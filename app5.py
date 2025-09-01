from flask import Flask, request, render_template, session, redirect, url_for, send_file
import pandas as pd
import io
from datetime import datetime, date, timedelta # Ensure timedelta is imported
import collections
# import calendar # Not used in the current current_version of calculate_trips_from_dataframe
import tempfile
import os
import threading
import webview
import sys
import time as time_module
# import json # Not directly used in calculate_trips_from_dataframe, but might be used elsewhere in Flask app

# --- CONFIGURATION ---
# Define the column names as they appear in your Excel file
COL_INVOICE_ID = 'INVOICE' # Retained as it's part of original global config
COL_VEHICLE_NO = 'TRUCK'
COL_INVOICE_DATE = 'D/DATE'
COL_INVOICE_TIME = 'D/TIME'
COL_CUSTOMER = 'CUSTOMER' # Can be None if not used

TRIP_START_HOUR = 7  # 7 AM

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_super_secret_key_replace_me_in_prod_12345') # Added a default for safety
FLASK_PORT = 5000 # You can change this if needed

# Global variable to store trip summary data as backup
trip_summary_backup = None

# --- Updated Helper Function for Trip Calculation (Datewise Logic) ---
def calculate_trips_from_dataframe(df_invoices, customer_filter=None):
    error_message = None
    # --- Preserve original column checks and initial setup ---
    required_columns = [COL_VEHICLE_NO, COL_INVOICE_DATE, COL_INVOICE_TIME]
    if COL_CUSTOMER: # Only add if COL_CUSTOMER is configured and not None
        required_columns.append(COL_CUSTOMER)

    missing_columns = [col for col in required_columns if col not in df_invoices.columns]
    if missing_columns:
        error_message = f"Missing required columns: {', '.join(missing_columns)}."
        # Return empty DataFrame and 0 trips, plus the error message
        return pd.DataFrame(), 0, error_message

    # --- ROBUST DATE/TIME CONVERSION ---
    try:
        # 1. Convert date and time columns to string to ensure clean concatenation
        date_str = df_invoices[COL_INVOICE_DATE].astype(str).str.split(' ').str[0] # Get only the date part
        time_str = df_invoices[COL_INVOICE_TIME].astype(str) # Convert time to string

        # 2. Combine date and time strings
        combined_datetime_str = date_str + ' ' + time_str

        # 3. Convert the combined string to a datetime object
        df_invoices['Invoice_DateTime'] = pd.to_datetime(combined_datetime_str, errors='coerce')

    except Exception as e:
        error_message = f"Error converting date/time columns. Ensure '{COL_INVOICE_DATE}' and '{COL_INVOICE_TIME}' are in expected formats. Error: {e}"
        return pd.DataFrame(), 0, error_message

    # Drop rows where Invoice_DateTime could not be parsed (became NaT)
    df = df_invoices.dropna(subset=['Invoice_DateTime']).copy() # Use .copy() to avoid SettingWithCopyWarning

    if df.empty:
        error_message = "No valid invoice data found after processing dates/times. Please check your date and time columns."
        return pd.DataFrame(), 0, error_message

    # Apply customer filter (if COL_CUSTOMER is defined and present)
    if customer_filter and COL_CUSTOMER and COL_CUSTOMER in df.columns:
        df = df[df[COL_CUSTOMER] == customer_filter]
    elif customer_filter and (not COL_CUSTOMER or COL_CUSTOMER not in df.columns):
        error_message = f"Customer filter '{customer_filter}' applied, but '{COL_CUSTOMER}' column not found or not configured."
        # Proceeding with unfiltered data if column is missing but filter was intended, error_message will be shown.

    if df.empty: # Check again after filtering
        error_message = "No data found for the selected filters." if customer_filter else "No invoice data available."
        return pd.DataFrame(), 0, error_message

    # Sort by truck and then by Invoice_DateTime to correctly identify trips
    df = df.sort_values(by=[COL_VEHICLE_NO, 'Invoice_DateTime']).reset_index(drop=True)

    # --- NEW DATEWISE LOGIC ---
    
    # Create trip_day for each invoice (7 AM to 6:59 AM next day)
    def get_trip_date(dt):
        if dt.hour < TRIP_START_HOUR:  # Before 7 AM
            # This belongs to the previous day's trip cycle
            return (dt - timedelta(days=1)).date()
        else:  # 7 AM or later
            # This belongs to current day's trip cycle
            return dt.date()
    
    df['Trip_Date'] = df['Invoice_DateTime'].apply(get_trip_date)
    
    # Get the overall date range from first to last trip date
    min_trip_date = df['Trip_Date'].min()
    max_trip_date = df['Trip_Date'].max()
    
    # Create list of all dates in range
    date_range = []
    current_date = min_trip_date
    while current_date <= max_trip_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Filter out trips that start before 7 AM on the very first date
    first_calendar_date = df['Invoice_DateTime'].dt.date.min()
    df = df[~((df['Invoice_DateTime'].dt.date == first_calendar_date) & 
             (df['Invoice_DateTime'].dt.hour < TRIP_START_HOUR))]
    
    if df.empty:
        error_message = "No valid trips found after applying time filters."
        return pd.DataFrame(), 0, error_message
    
    # Recalculate trip dates after filtering
    df['Trip_Date'] = df['Invoice_DateTime'].apply(get_trip_date)
    
    # Initialize trip counts dictionary: {truck_id: {date: count}}
    truck_trip_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    
    # Get all unique trucks
    trucks = sorted(df[COL_VEHICLE_NO].unique())
    
    # Calculate trips for each truck
    for truck_id in trucks:
        truck_df = df[df[COL_VEHICLE_NO] == truck_id].copy()
        truck_df = truck_df.sort_values('Invoice_DateTime').reset_index(drop=True)
        
        if truck_df.empty:
            continue
            
        # Track current trip for this truck
        current_trip_start_time = None
        current_trip_date = None
        
        for index, row in truck_df.iterrows():
            invoice_datetime = row['Invoice_DateTime']
            trip_date = row['Trip_Date']
            
            if current_trip_start_time is None:
                # First invoice for this truck - starts a new trip
                current_trip_start_time = invoice_datetime
                current_trip_date = trip_date
            else:
                # Check if this invoice is more than 1 hour after the current trip start
                if (invoice_datetime - current_trip_start_time) > timedelta(hours=1): # Changed from hours=1, minutes=30 to hours=1
                    # Previous trip is complete - count it
                    truck_trip_counts[truck_id][current_trip_date] += 1
                    
                    # Start new trip with current invoice
                    current_trip_start_time = invoice_datetime
                    current_trip_date = trip_date
                # else: current invoice is part of the same trip (within 1 hour)
        
        # Don't forget to count the last trip for this truck
        if current_trip_start_time is not None and current_trip_date is not None:
            truck_trip_counts[truck_id][current_trip_date] += 1
    
    # Build the result DataFrame
    result_data = []
    total_all_trips = 0
    
    for i, truck_id in enumerate(trucks, 1):
        row_data = {'S.No': i, 'Truck ID': truck_id}
        truck_total = 0
        
        # Add columns for each date in range
        for trip_date in date_range:
            date_str = trip_date.strftime('%d %b')  # Format: "01 Jan"
            trip_count = truck_trip_counts[truck_id][trip_date]
            row_data[date_str] = trip_count
            truck_total += trip_count
        
        row_data['Total Trips'] = truck_total
        total_all_trips += truck_total
        result_data.append(row_data)
    
    # Create DataFrame
    trip_summary_df = pd.DataFrame(result_data)
    
    # Ensure columns are in correct order
    if not trip_summary_df.empty:
        base_columns = ['S.No', 'Truck ID']
        date_columns = [trip_date.strftime('%d %b') for trip_date in date_range]
        final_columns = base_columns + date_columns + ['Total Trips']
        
        # Reorder columns
        trip_summary_df = trip_summary_df[final_columns]
    
    return trip_summary_df, total_all_trips, error_message

# --- Function to clean up temporary files ---
def cleanup_temp_files():
    """Clean up temporary files from session more robustly."""
    paths_to_clean = []
    try:
        if 'dataframe_path' in session and session['dataframe_path']:
            paths_to_clean.append(session.pop('dataframe_path'))
        if 'trip_summary_path' in session and session['trip_summary_path']:
            paths_to_clean.append(session.pop('trip_summary_path'))
    except RuntimeError: # Occurs if called outside request context (e.g., on app shutdown)
        print("Cleanup: No request context, session not accessed directly.")
        # Potentially add logic here to clean up known temp file locations if needed
        # For now, this means session-based cleanup won't run on app_teardown if no active request.
        return 


    for path in paths_to_clean:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                print(f"Successfully removed temp file: {path}")
            except OSError as e:
                print(f"Warning: Could not remove temp file {path}: {e}")
        elif path:
            print(f"Warning: Temp file path found in session but file does not exist: {path}")


# --- Flask Routes (largely from original, with minor adjustments for clarity/robustness) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global trip_summary_backup # To store a copy of the summary DataFrame

    trip_summary_html = None
    total_trips = 0
    customer_options = []
    selected_customer = request.form.get('customer_filter', '')
    error = None
    trip_summary_df = pd.DataFrame()

    df_raw = pd.DataFrame()
    overall_first_invoice_date = None # Initialize
    overall_last_invoice_date = None  # Initialize

    try:
        # Attempt to load DataFrame from session if available
        if 'dataframe_path' in session:
            temp_file_path = session['dataframe_path']
            if os.path.exists(temp_file_path):
                try:
                    df_raw = pd.read_parquet(temp_file_path)
                    if not df_raw.empty:
                        if COL_INVOICE_DATE in df_raw.columns and COL_INVOICE_TIME in df_raw.columns:
                            try:
                                temp_date_str = df_raw[COL_INVOICE_DATE].astype(str).str.split(' ').str[0]
                                temp_time_str = df_raw[COL_INVOICE_TIME].astype(str)
                                temp_combined_datetime = pd.to_datetime(temp_date_str + ' ' + temp_time_str, errors='coerce')
                                overall_first_invoice_date = temp_combined_datetime.min()
                                overall_last_invoice_date = temp_combined_datetime.max()
                            except Exception as e:
                                print(f"Warning: Could not determine overall date range from df_raw: {e}")
                                overall_first_invoice_date = None
                                overall_last_invoice_date = None

                        if COL_CUSTOMER and COL_CUSTOMER in df_raw.columns:
                            customer_options = sorted(df_raw[COL_CUSTOMER].dropna().unique().tolist())
                except Exception as e:
                    error = f"Session data (Parquet) seems corrupted or invalid. Please re-upload. Details: {e}"
                    cleanup_temp_files()
            else:
                error = "Session file not found. Please re-upload your Excel file."
                cleanup_temp_files()

        if request.method == 'POST':
            if 'file' in request.files and request.files['file'].filename != '':
                cleanup_temp_files()
                file = request.files['file']
                if file and file.filename.endswith(('.xlsx', '.xls')):
                    try:
                        file_bytes = io.BytesIO(file.read())
                        df_raw = pd.read_excel(file_bytes, engine=None)

                        fd, temp_file_path = tempfile.mkstemp(suffix='.parquet')
                        os.close(fd)
                        df_raw.to_parquet(temp_file_path, index=False)
                        session['dataframe_path'] = temp_file_path

                        if COL_CUSTOMER and COL_CUSTOMER in df_raw.columns:
                            customer_options = sorted(df_raw[COL_CUSTOMER].dropna().unique().tolist())
                        selected_customer = ''
                        error = None

                        # After new upload, set overall dates
                        if COL_INVOICE_DATE in df_raw.columns and COL_INVOICE_TIME in df_raw.columns:
                            try:
                                temp_date_str = df_raw[COL_INVOICE_DATE].astype(str).str.split(' ').str[0]
                                temp_time_str = df_raw[COL_INVOICE_TIME].astype(str)
                                temp_combined_datetime = pd.to_datetime(temp_date_str + ' ' + temp_time_str, errors='coerce')
                                overall_first_invoice_date = temp_combined_datetime.min()
                                overall_last_invoice_date = temp_combined_datetime.max()
                            except Exception as e:
                                print(f"Warning: Could not determine overall date range from new upload: {e}")
                                overall_first_invoice_date = None
                                overall_last_invoice_date = None


                    except Exception as e:
                        error = f"Error processing Excel file: {e}. Ensure it's a valid Excel format."
                        cleanup_temp_files()
                else:
                    error = "Invalid file type. Please upload an Excel (.xlsx or .xls) file."
                    cleanup_temp_files()

            elif 'dataframe_path' not in session or df_raw.empty:
                if not error:
                    error = "No data loaded. Please upload an Excel file to begin."

            if error is None and not df_raw.empty:
                trip_summary_df, total_trips, calc_error = calculate_trips_from_dataframe(
                    df_raw.copy(),
                    customer_filter=selected_customer if selected_customer else None
                )
                if calc_error:
                    error = calc_error
                elif trip_summary_df.empty:
                    if not error:
                         error = "No trips found for the selected criteria."
                else:
                    trip_summary_html = trip_summary_df.to_html(
                        classes='table table-striped table-bordered table-hover table-sm',
                        index=False,
                        table_id='trip-summary-table',
                        escape=False
                    )

                    try:
                        fd_summary, temp_summary_path = tempfile.mkstemp(suffix='.parquet')
                        os.close(fd_summary)
                        trip_summary_df.to_parquet(temp_summary_path, index=False)
                        session['trip_summary_path'] = temp_summary_path
                        trip_summary_backup = trip_summary_df.copy()
                    except Exception as e:
                        print(f"Warning: Could not store trip summary to Parquet: {e}")
                        try:
                            session['trip_summary_json'] = trip_summary_df.to_json(orient='records')
                            trip_summary_backup = trip_summary_df.copy()
                        except Exception as e_json:
                             print(f"Warning: Could not store trip summary as JSON: {e_json}")
                             trip_summary_backup = trip_summary_df.copy()

    except Exception as e:
        print(f"CRITICAL ERROR IN INDEX ROUTE: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        error = f"An unexpected application error occurred: {e}. Please try again or check logs."
        customer_options = []
        total_trips = 0
        trip_summary_html = None
        cleanup_temp_files()

    return render_template('index.html',
                           trip_summary_html=trip_summary_html,
                           total_trips=total_trips,
                           customer_options=customer_options,
                           selected_customer=selected_customer,
                           error=error,
                           overall_first_invoice_date=overall_first_invoice_date, # Pass this
                           overall_last_invoice_date=overall_last_invoice_date) # Pass this

@app.route('/export_excel')
def export_excel():
    global trip_summary_backup
    
    trip_summary_df_export = None # Use a different variable name for clarity
    
    # Try multiple methods to retrieve the trip summary data for export
    try:
        # Method 1: Try to load from parquet file path in session
        if 'trip_summary_path' in session:
            temp_summary_path = session['trip_summary_path']
            if os.path.exists(temp_summary_path):
                try:
                    trip_summary_df_export = pd.read_parquet(temp_summary_path)
                    # print("Export: Loaded trip summary from session Parquet file.")
                except Exception as e:
                    print(f"Export Error: Failed to read Parquet from session path '{temp_summary_path}': {e}")
        
        # Method 2: Try to load from JSON in session (if Parquet failed or path not found)
        if trip_summary_df_export is None and 'trip_summary_json' in session:
            try:
                json_data = session['trip_summary_json']
                trip_summary_df_export = pd.read_json(json_data, orient='records')
                # print("Export: Loaded trip summary from session JSON.")
            except Exception as e:
                 print(f"Export Error: Failed to read JSON from session: {e}")
        
        # Method 3: Use global backup (if session methods failed)
        if trip_summary_df_export is None and trip_summary_backup is not None:
            trip_summary_df_export = trip_summary_backup.copy() # Use a copy
            # print("Export: Loaded trip summary from global backup.")
            
        if trip_summary_df_export is None or trip_summary_df_export.empty:
            print("Export ERROR: No trip summary data available for export.")
            # Redirect to home with an error message if more user-friendly
            return "No trip summary data available for export. Please calculate trips first.", 404

        # Create a proper filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'truck_trip_summary_datewise_{timestamp}.xlsx'

        output_excel = io.BytesIO()
        try:
            with pd.ExcelWriter(output_excel, engine='openyxl') as writer:
                trip_summary_df_export.to_excel(writer, index=False, sheet_name='Trip Summary')
                
                # Auto-adjust column widths (optional, but good for usability)
                worksheet = writer.sheets['Trip Summary']
                for column_cells in worksheet.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter # Get column letter
                    for cell in column_cells:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2) * 1.2 # Factor for better fitting
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Add a title row (optional)
                # worksheet.insert_rows(1)
                # worksheet['A1'] = f'Truck Trip Summary - Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            
            output_excel.seek(0)
            # print(f"Excel file created for download. Size: {len(output_excel.getvalue())} bytes")

        except Exception as e:
            print(f"Error creating Excel file for export: {e}. Falling back to CSV.")
            # Fallback to CSV if Excel creation fails
            output_csv = io.StringIO()
            trip_summary_df_export.to_csv(output_csv, index=False)
            output_csv.seek(0)
            filename = filename.replace('.xlsx', '.csv')
            
            return send_file(
                io.BytesIO(output_csv.getvalue().encode('utf-8')), # Ensure encoding
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )

        return send_file(
            output_excel,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"General Export error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return f"Error exporting file: {e}. Please check server logs.", 500

# Clean up temporary files on app context teardown
@app.teardown_appcontext
def cleanup_on_teardown(exception=None): # Add exception argument
    # print("App context teardown: Attempting to clean up temp files.")
    cleanup_temp_files() # Call the refined cleanup function

# --- Webview specific part (conditionally imported and used) ---
# Function to start Flask in a separate thread for desktop application
def start_flask():
    print(f"Flask server starting on http://127.0.0.1:{FLASK_PORT}/", file=sys.stderr)
    # Set debug=False for production/webview use, use_reloader=False is important for threaded mode
    app.run(host='127.0.0.1', port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

if __name__ == '__main__':
    # Check if running in a bundled executable (e.g., PyInstaller)
    # This helps decide if webview should be used or just run as a web server.
    is_frozen = getattr(sys, 'frozen', False)

    if is_frozen or '--webview' in sys.argv: # Use webview if bundled or explicitly requested
        try:
            import webview # Import here to avoid error if not installed and not needed
            
            # Start Flask in a separate thread
            flask_thread = threading.Thread(target=start_flask)
            flask_thread.daemon = True # Allows main program to exit even if thread is running
            flask_thread.start()

            # Give Flask a moment to start up
            time_module.sleep(2) # Adjust if needed

            # Create webview window
            webview.create_window(
                'Truck Trip Calculator',
                f'http://127.0.0.1:{FLASK_PORT}/',
                width=1200,
                height=800,
                min_size=(800, 600),
                resizable=True,
                maximized=False, # Start maximized but not fullscreen
                confirm_close=True # Ask user before closing
            )
            webview.start(debug=False) # http_server=True by default if serving local files, not needed here
            
            # After webview.start() returns (window closed), ensure Flask thread can terminate
            # This might involve signaling Flask to shut down if it doesn't stop automatically.
            # For simple cases, daemon thread might be enough.
            print("Webview window closed. Application shutting down.")

        except ImportError:
            print("Webview library not found. Please install it (`pip install pywebview`) to run as a desktop app.")
            print("Falling back to running as a standard web server. Access at http://127.0.0.1:5000/")
            start_flask() # Run Flask normally if webview import fails
        except Exception as e:
            print(f"Error starting webview: {e}")
            print("Falling back to running as a standard web server. Access at http://127.0.0.1:5000/")
            start_flask() # Run Flask normally on other webview errors
    else:
        # Run as a standard web server if not bundled and --webview not specified
        print("Running as a standard web server. Access at http://127.0.0.1:5000/")
        print("To run as a desktop app with webview, try: python app.py --webview (if pywebview is installed)")
        start_flask()
