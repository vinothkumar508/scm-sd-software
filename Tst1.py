import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QTabWidget, QTableWidget, QTableWidgetItem, QTableView, QLabel, QHBoxLayout,
    QComboBox, QDateEdit, QGroupBox, QFormLayout, QMessageBox, QLineEdit,
    QStyleFactory, QFrame, QRadioButton, QScrollArea, QListWidget, QCheckBox
)
from PyQt5.QtCore import QAbstractTableModel, Qt, QDate
from PyQt5.QtGui import QIcon, QFont, QPixmap
import sqlite3
import os

EXPECTED_HEADERS = [
    'Date', 'Particulars', 'Voucher Type', 'Voucher No.', 'Voucher Ref. No.',
    'Delivery Note No. & Date', 'Despatch Doc. No', 'Despatch Through',
    'Destination', 'Shipping No.', 'Shipping Date', 'Quantity', 'Value',
    'Gross Total', 'GST SALES @ 28 %', 'OUTPUT IGST 28%', 'IGST SALES @ 28 %', 'Customer'
]

REQUIRED_HEADERS = [
    'Date', 'Particulars', 'Voucher No.', 'Despatch Through', 'Destination',
    'Quantity', 'Value', 'Gross Total', 'Customer'
]

INVOICE_HEADER = 'HONDA MOTORCYCLE AND SCOOTER INDIA PRIVATE LIMITED'

OUTPUT_ORDER = [
    "Date", "Voucher No.", "Despatch Through", "Particulars", "Quantity", "Value", "Gross Total", "Customer"
]

# Add a global stylesheet for modern look
APP_STYLESHEET = """
QMainWindow { background-color: #f8fafc; }
QTabWidget::pane { border: 1px solid #cbd5e1; border-radius: 12px; background: #ffffff; margin-top: 8px; }
QTabBar::tab { background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 8px 8px 0 0; padding: 12px 24px;
font-size: 15px; font-weight: 600; margin-right: 4px; min-width: 120px; color: #475569; }
QTabBar::tab:selected { background: #ffffff; color: #1e40af; border-bottom: 3px solid #1e40af; font-weight: 700; }
QTabBar::tab:hover { background: #f1f5f9; color: #1e40af; }
QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a8a, stop:1 #2563eb); color: #ffffff;
border: none; border-radius: 8px; padding: 14px 28px; font-size: 15px; font-weight: bold; min-height: 24px; }
QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #1e40af); }
QPushButton:pressed { background: #1e40af; }
QGroupBox { border: 2px solid #e2e8f0; border-radius: 12px; margin-top: 16px;
background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); font-size: 16px; font-weight: 700;
color: #1e293b; padding-top: 16px; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 12px 0 12px; background: #ffffff; border-radius: 6px; }
QLabel { font-size: 14px; color: #374151; }
QTableView, QTableWidget { border: 2px solid #e2e8f0; border-radius: 12px; background: #ffffff; font-size: 13px;
selection-background-color: #dbeafe; gridline-color: #f1f5f9; alternate-background-color: #f8fafc; }
QTableView::item, QTableWidget::item { padding: 8px; border-bottom: 1px solid #f1f5f9; }
QTableView::item:selected, QTableWidget::item:selected { background-color: #dbeafe; color: #1e293b; }
QComboBox, QDateEdit, QLineEdit { border: 2px solid #e2e8f0; border-radius: 8px; padding: 8px 12px;
font-size: 14px; background: #ffffff; color: #374151; min-height: 20px; }
QComboBox:hover, QDateEdit:hover, QLineEdit:hover { border-color: #3b82f6; }
QComboBox:focus, QDateEdit:focus, QLineEdit:focus { border-color: #1e40af; background: #f8fafc; }
QHeaderView::section { background: #1e40af; color: #ffffff; font-weight: bold; font-size: 15px;
border: none; padding: 14px 8px; border-right: 2px solid #2563eb; }
QHeaderView::section:last { border-right: none; }
QListWidget { border: 2px solid #e2e8f0; border-radius: 8px; background: #ffffff; font-size: 13px; }
QListWidget::item { padding: 5px; border-bottom: 1px solid #f1f5f9; }
QListWidget::item:selected { background-color: #dbeafe; color: #1e293b; }
"""

# Helper functions
def map_columns(df, expected_cols):
    actual_cols = list(df.columns)
    mapping = {}
    for exp in expected_cols:
        found = None
        for act in actual_cols:
            if str(act).replace(' ', '').lower() == exp.replace(' ', '').lower():
                found = act
                break
        mapping[exp] = found
    return mapping

def find_header_row(raw_df, expected_headers, min_matches=4):
    for i in range(min(20, len(raw_df))):
        row = raw_df.iloc[i].fillna("").astype(str).tolist()
        matches = 0
        for cell in row:
            for exp in expected_headers:
                if cell.replace(' ', '').lower() == exp.replace(' ', '').lower():
                    matches += 1
                    break
        if matches >= min_matches:
            return i
    return 0

def convert_to_numeric(value):
    if pd.isna(value) or value == "":
        return None
    try:
        cleaned = str(value).replace(',', '').replace('₹', '').replace('$', '').strip()
        return float(cleaned)
    except:
        return None

def extract_and_clean_data(df, manual_map=None):
    col_map = map_columns(df, EXPECTED_HEADERS)
    if manual_map:
        for k, v in manual_map.items():
            if v != '':
                col_map[k] = v

    cleaned_rows = []
    for idx, row in df.iterrows():
        cleaned_row = {}
        for exp in EXPECTED_HEADERS:
            src_col = col_map.get(exp)
            cleaned_row[exp] = row[src_col] if src_col in row else None
        cleaned_rows.append(cleaned_row)

    cleaned_df = pd.DataFrame(cleaned_rows)
    cleaned_df = cleaned_df.replace("", pd.NA).ffill()

    if 'Voucher No.' in cleaned_df.columns:
        def is_sipl_voucher(val):
            if pd.isna(val) or val == "":
                return False
            return str(val).strip().upper().startswith('SIPL')
        cleaned_df = cleaned_df[cleaned_df['Voucher No.'].apply(is_sipl_voucher)]

    numeric_columns = [
        'Quantity', 'Value', 'Gross Total',
        'GST SALES @ 28 %', 'OUTPUT IGST 28%', 'IGST SALES @ 28 %'
    ]
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(convert_to_numeric)

    cleaned_df = cleaned_df[[col for col in REQUIRED_HEADERS if col in cleaned_df.columns]]

    if 'Date' in cleaned_df.columns:
        cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

    def is_grand_total(val):
        return str(val).strip().lower() == 'grand total'
    cleaned_df = cleaned_df[~cleaned_df['Particulars'].apply(is_grand_total)]

    return cleaned_df, col_map

def reorder_columns(df):
    cols = [col for col in OUTPUT_ORDER if col in df.columns]
    return df[cols]

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

# DB functions - Updated with PO Number and Color
def init_master_db(db_path='master_data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS master_data (
            part_number VARCHAR PRIMARY KEY,
            model TEXT,
            part_category TEXT,
            part_supply TEXT,
            po_number TEXT,
            color TEXT,
            normalized_part VARCHAR
        )
    ''')
    conn.commit()
    conn.close()

def get_db_connection(db_path='master_data.db'):
    return sqlite3.connect(db_path)

def add_master_record(part_number, model, part_category, part_supply, po_number='', color='', db_path='master_data.db'):
    normalized_part = str(part_number).replace(' ', '').upper()
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO master_data (part_number, model, part_category, part_supply, po_number, color, normalized_part)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (part_number, model, part_category, part_supply, po_number, color, normalized_part))
    conn.commit()
    conn.close()

def update_master_record(part_number, model, part_category, part_supply, po_number='', color='', db_path='master_data.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE master_data SET model=?, part_category=?, part_supply=?, po_number=?, color=? WHERE part_number=?
    ''', (model, part_category, part_supply, po_number, color, part_number))
    conn.commit()
    conn.close()

def delete_master_record(part_number, db_path='master_data.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM master_data WHERE part_number=?', (part_number,))
    conn.commit()
    conn.close()

def fetch_all_master_records(db_path='master_data.db'):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT part_number, model, part_category, part_supply, po_number, color, normalized_part FROM master_data')
    records = cursor.fetchall()
    conn.close()
    return records

class DataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)

        self.load_button = QPushButton('Load Excel File')
        self.load_button.setToolTip('Select an Excel file to load sales data')
        self.load_button.setStyleSheet("font-size: 15px; padding: 10px 20px; background: #1976d2; color: #fff; border-radius: 8px;")
        self.load_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_button, alignment=Qt.AlignTop)

        self.banner = QLabel('📊 Load your Excel file and view the data below. Master data integration with PO numbers and colors.')
        self.banner.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.banner.setStyleSheet("background-color: #e3f2fd; border-radius: 10px; padding: 15px; font-size: 14px; color: #1a237e;")
        self.layout.addWidget(self.banner)

        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: #f9f9f9; background-color: #ffffff; font-size: 13px; border-radius: 8px;")
        self.layout.addWidget(self.table, stretch=1)

        export_layout = QHBoxLayout()
        export_layout.setSpacing(10)
        export_layout.setContentsMargins(0, 10, 0, 0)

        self.export_excel_btn = QPushButton('Export to Excel')
        self.export_excel_btn.setToolTip('Export data to Excel')
        self.export_csv_btn = QPushButton('Export to CSV')
        self.export_csv_btn.setToolTip('Export data to CSV')

        self.export_excel_btn.clicked.connect(self.export_excel)
        self.export_csv_btn.clicked.connect(self.export_csv)

        export_layout.addWidget(self.export_excel_btn)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addStretch()
        self.layout.addLayout(export_layout)

        self.summary_bar = QLabel()
        self.summary_bar.setStyleSheet("background: #e0f2f7; border-radius: 8px; padding: 10px 18px; font-size: 13px; margin-top: 8px; font-weight: bold; color: #006064;")
        self.layout.addWidget(self.summary_bar)

        self.setLayout(self.layout)

        self.df = pd.DataFrame()
        self.cleaned_df = pd.DataFrame()
        self.col_map = None
        self.header_row = 0
        self.master_df = self.load_master_data()
        self.parent_tabs = None

    def load_master_data(self):
        records = fetch_all_master_records()
        return pd.DataFrame(records, columns=["Part Number", "Model", "Part Category", "Part Supply", "PO Number", "Color", "Normalized Part"])

    def join_with_master(self, df):
        if df.empty or self.master_df.empty:
            return df

        df = df.copy()
        master_df = self.master_df.copy()

        df['__normalized_part'] = df['Particulars'].astype(str).str.replace(' ', '').str.upper()
        joined = df.merge(master_df, left_on='__normalized_part', right_on='Normalized Part', how='left', suffixes=('', '_master'))
        joined.drop(columns=['__normalized_part', 'Normalized Part'], inplace=True)

        # CONDITIONAL: OE to OE SPD based on destination
        if 'Destination' in joined.columns and 'Part Supply' in joined.columns:
            spd_condition = joined['Destination'].astype(str).str.contains(
                r'honda\s*-?\s*spd', case=False, na=False, regex=True
            )
            oe_condition = joined['Part Supply'].astype(str).str.upper() == 'OE'
            joined.loc[spd_condition & oe_condition, 'Part Supply'] = 'OE SPD'

        return joined

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '', 'Excel Files (*.xlsx *.xls)')
        if file_name:
            raw_df = pd.read_excel(file_name, dtype=str, header=None)
            self.header_row = find_header_row(raw_df, EXPECTED_HEADERS, min_matches=4)
            df = pd.read_excel(file_name, dtype=str, header=self.header_row)
            self.df = df.fillna("")

            if 'Particulars' in self.df.columns:
                self.df['Particulars'] = self.df['Particulars'].astype(str).str.strip()

            if 'Timestamp' in self.df.columns:
                self.df = self.df.drop(columns=['Timestamp'])

            self.cleaned_df, self.col_map = extract_and_clean_data(self.df)
            self.master_df = self.load_master_data()
            self.cleaned_df = self.join_with_master(self.cleaned_df)

            for col in ["Model", "Part Category", "Part Supply", "PO Number", "Color"]:
                if col not in self.cleaned_df.columns:
                    self.cleaned_df[col] = ''

            if 'Part Supply' in self.cleaned_df.columns:
                self.cleaned_df.loc[:, 'Part Supply'] = self.cleaned_df['Part Supply'].ffill().bfill().fillna('')
            else:
                self.cleaned_df['Part Supply'] = ''

            self.update_table()

            # Update other tabs if they exist
            if self.parent_tabs:
                if hasattr(self.parent_tabs, 'part_tab'):
                    self.parent_tabs.part_tab.update_table(self.cleaned_df)
                if hasattr(self.parent_tabs, 'header_tab'):
                    self.parent_tabs.header_tab.update_table(self.cleaned_df)
                if hasattr(self.parent_tabs, 'visualization_tab'):
                    self.parent_tabs.visualization_tab.update_data(self.cleaned_df)

    def update_table(self):
        if not self.cleaned_df.empty:
            df = self.cleaned_df.copy()
            if 'Timestamp' in df.columns:
                df = df.drop(columns=['Timestamp'])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

            model = PandasModel(df)
            self.table.setModel(model)
            self.auto_fit_columns()
            self.update_summary(df)
        else:
            self.table.setModel(PandasModel(pd.DataFrame()))
            self.update_summary(pd.DataFrame())

    def auto_fit_columns(self):
        header = self.table.horizontalHeader()
        for i in range(self.table.model().columnCount()):
            header.setSectionResizeMode(i, header.ResizeToContents)

    def update_summary(self, df):
        if not df.empty:
            total_rows = len(df)
            gross_total = df['Gross Total'].sum() if 'Gross Total' in df.columns else 0
            self.summary_bar.setText(f'Total Records: {total_rows} | Gross Total: ₹{gross_total:,.2f}')
        else:
            self.summary_bar.setText('No data loaded')

    def export_excel(self):
        df = self.cleaned_df.copy()
        if not df.empty:
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save Excel File', '', 'Excel Files (*.xlsx)')
            if file_name:
                df.to_excel(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Data exported to Excel successfully!')

    def export_csv(self):
        df = self.cleaned_df.copy()
        if not df.empty:
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')
            if file_name:
                df.to_csv(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Data exported to CSV successfully!')

class PartTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)

        # Filter Group Box
        filter_group_box = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        filter_layout.setContentsMargins(10, 15, 10, 10)
        filter_style = "font-size: 12px; min-height: 22px; padding: 0 4px; min-width: 60px; max-width: 100px;"

        # From Date
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setDisplayFormat('yyyy-MM-dd')
        self.from_date.setStyleSheet(filter_style)
        from_label = QLabel('From Date:')
        filter_layout.addWidget(from_label)
        filter_layout.addWidget(self.from_date)

        # To Date
        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setDisplayFormat('yyyy-MM-dd')
        self.to_date.setStyleSheet(filter_style)
        to_label = QLabel('To Date:')
        filter_layout.addWidget(to_label)
        filter_layout.addWidget(self.to_date)

        # Part Supply Radio Buttons
        part_label = QLabel('Part Supply:')
        filter_layout.addWidget(part_label)
        self.oe_radio = QRadioButton('OE')
        self.spd_radio = QRadioButton('SPD')
        self.oe_spd_radio = QRadioButton('OE SPD')
        self.all_radio = QRadioButton('All')
        self.all_radio.setChecked(True)

        for rb in [self.all_radio, self.oe_radio, self.spd_radio, self.oe_spd_radio]:
            filter_layout.addWidget(rb)

        filter_layout.addStretch()
        filter_group_box.setLayout(filter_layout)
        self.layout.addWidget(filter_group_box)

        # Title
        title_label = QLabel('🔧 Showing all part numbers with PO numbers and colors (excluding invoice headers)')
        title_label.setStyleSheet("background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 10px; padding: 15px; border: 1px solid #90caf9; margin-top: 10px;")
        self.layout.addWidget(title_label)

        # Table
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: #f8fafc; background-color: #ffffff; font-size: 14px; border-radius: 8px;")
        self.layout.addWidget(self.table, stretch=1)

        # Export Buttons
        export_layout = QHBoxLayout()
        export_layout.setSpacing(10)
        export_layout.setContentsMargins(0, 10, 0, 0)
        self.export_excel_btn = QPushButton('Export to Excel')
        self.export_csv_btn = QPushButton('Export to CSV')
        self.export_excel_btn.clicked.connect(self.export_excel)
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_excel_btn)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addStretch()
        self.layout.addLayout(export_layout)

        self.setLayout(self.layout)
        self.current_df = pd.DataFrame()

        # Connect filter signals
        self.from_date.dateChanged.connect(self.apply_filters)
        self.to_date.dateChanged.connect(self.apply_filters)
        self.oe_radio.toggled.connect(self.apply_filters)
        self.spd_radio.toggled.connect(self.apply_filters)
        self.oe_spd_radio.toggled.connect(self.apply_filters)
        self.all_radio.toggled.connect(self.apply_filters)

    def auto_fit_columns(self):
        header = self.table.horizontalHeader()
        for i in range(self.table.model().columnCount()):
            header.setSectionResizeMode(i, header.ResizeToContents)

    def update_table(self, df=None):
        if df is not None:
            data_df = df.copy()
        else:
            parent_tabs = getattr(self, 'parent_tabs', None)
            data_df = None
            if parent_tabs and hasattr(parent_tabs, 'data_tab'):
                data_df = parent_tabs.data_tab.cleaned_df.copy()

        if data_df is not None and not data_df.empty:
            part_table = data_df[data_df['Particulars'] != INVOICE_HEADER]
            if 'Timestamp' in part_table.columns:
                part_table = part_table.drop(columns=['Timestamp'])
            if 'Date' in part_table.columns:
                part_table.loc[:, 'Date'] = pd.to_datetime(part_table['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

            self.current_df = part_table

            if 'Date' in part_table.columns and not part_table['Date'].isnull().all():
                min_date = pd.to_datetime(part_table['Date'], errors='coerce').min()
                max_date = pd.to_datetime(part_table['Date'], errors='coerce').max()
                if pd.notnull(min_date):
                    min_qdate = QDate(min_date.year, min_date.month, min_date.day)
                    self.from_date.setMinimumDate(min_qdate)
                    self.to_date.setMinimumDate(min_qdate)
                    self.from_date.setDate(min_qdate)
                if pd.notnull(max_date):
                    max_qdate = QDate(max_date.year, max_date.month, max_date.day)
                    self.from_date.setMaximumDate(max_qdate)
                    self.to_date.setMaximumDate(max_qdate)
                    self.to_date.setDate(max_qdate)

            self.apply_filters()
        else:
            self.table.setModel(PandasModel(pd.DataFrame()))
            self.current_df = pd.DataFrame()

    def apply_filters(self):
        df = self.current_df.copy()

        # Date filter
        if 'Date' in df.columns:
            from_date = self.from_date.date().toPyDate()
            to_date = self.to_date.date().toPyDate()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[(df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))]

        # Part Supply filter
        if 'Part Supply' in df.columns:
            if self.oe_radio.isChecked():
                df = df[df['Part Supply'] == 'OE']
            elif self.spd_radio.isChecked():
                df = df[df['Part Supply'] == 'SPD']
            elif self.oe_spd_radio.isChecked():
                df = df[df['Part Supply'] == 'OE SPD']

        model = PandasModel(df)
        self.table.setModel(model)
        self.auto_fit_columns()

    def export_excel(self):
        if not self.current_df.empty:
            df = self.current_df.copy()
            if 'Timestamp' in df.columns:
                df = df.drop(columns=['Timestamp'])
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save Excel File', '', 'Excel Files (*.xlsx)')
            if file_name:
                df.to_excel(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Part numbers exported to Excel successfully!')

    def export_csv(self):
        if not self.current_df.empty:
            df = self.current_df.copy()
            if 'Timestamp' in df.columns:
                df = df.drop(columns=['Timestamp'])
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')
            if file_name:
                df.to_csv(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Part numbers exported to CSV successfully!')

class HeaderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)

        # Filter Group Box
        filter_group_box = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        filter_layout.setContentsMargins(10, 15, 10, 10)
        filter_style = "font-size: 12px; min-height: 22px; padding: 0 4px; min-width: 60px; max-width: 100px;"

        # From Date
        self.from_date = QDateEdit()
        self.from_date.setCalendarPopup(True)
        self.from_date.setDisplayFormat('yyyy-MM-dd')
        self.from_date.setStyleSheet(filter_style)
        from_label = QLabel('From Date:')
        filter_layout.addWidget(from_label)
        filter_layout.addWidget(self.from_date)

        # To Date
        self.to_date = QDateEdit()
        self.to_date.setCalendarPopup(True)
        self.to_date.setDisplayFormat('yyyy-MM-dd')
        self.to_date.setStyleSheet(filter_style)
        to_label = QLabel('To Date:')
        filter_layout.addWidget(to_label)
        filter_layout.addWidget(self.to_date)

        # Part Supply Radio Buttons
        part_label = QLabel('Part Supply:')
        filter_layout.addWidget(part_label)
        self.oe_radio = QRadioButton('OE')
        self.spd_radio = QRadioButton('SPD')
        self.oe_spd_radio = QRadioButton('OE SPD')
        self.all_radio = QRadioButton('All')
        self.all_radio.setChecked(True)

        for rb in [self.all_radio, self.oe_radio, self.spd_radio, self.oe_spd_radio]:
            filter_layout.addWidget(rb)

        filter_layout.addStretch()
        filter_group_box.setLayout(filter_layout)
        self.layout.addWidget(filter_group_box)

        # Title
        title_label = QLabel('📄 Showing all invoice header records')
        title_label.setStyleSheet("background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 10px; padding: 15px; border: 1px solid #90caf9; margin-top: 10px;")
        self.layout.addWidget(title_label)

        # Table
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: #f8fafc; background-color: #ffffff; font-size: 14px; border-radius: 8px;")
        self.layout.addWidget(self.table, stretch=1)

        # Export Buttons
        export_layout = QHBoxLayout()
        export_layout.setSpacing(10)
        export_layout.setContentsMargins(0, 10, 0, 0)
        self.export_excel_btn = QPushButton('Export to Excel')
        self.export_csv_btn = QPushButton('Export to CSV')
        self.export_excel_btn.clicked.connect(self.export_excel)
        self.export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(self.export_excel_btn)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addStretch()
        self.layout.addLayout(export_layout)

        self.setLayout(self.layout)
        self.current_df = pd.DataFrame()

        # Connect filter signals
        self.from_date.dateChanged.connect(self.apply_filters)
        self.to_date.dateChanged.connect(self.apply_filters)
        self.oe_radio.toggled.connect(self.apply_filters)
        self.spd_radio.toggled.connect(self.apply_filters)
        self.oe_spd_radio.toggled.connect(self.apply_filters)
        self.all_radio.toggled.connect(self.apply_filters)

    def auto_fit_columns(self):
        header = self.table.horizontalHeader()
        for i in range(self.table.model().columnCount()):
            header.setSectionResizeMode(i, header.ResizeToContents)

    def update_table(self, df):
        if not df.empty:
            header_table = df[df['Particulars'] == INVOICE_HEADER].copy()
            for col in ["Customer", "Model", "Part Category", "Particulars", "Part Number", "Timestamp", "Time", "DateTime"]:
                if col in header_table.columns:
                    header_table.drop(columns=[col], inplace=True)

            if 'Date' in header_table.columns:
                header_table.loc[:, 'Date'] = pd.to_datetime(header_table['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

            # Get Part Supply from data view
            parent_tabs = getattr(self, 'parent_tabs', None)
            data_df = None
            if parent_tabs and hasattr(parent_tabs, 'data_tab'):
                data_df = parent_tabs.data_tab.cleaned_df.copy()

            if data_df is not None:
                data_df = data_df[data_df['Particulars'] != INVOICE_HEADER]
                if 'Voucher No.' in header_table.columns:
                    header_table.loc[:, 'Voucher No.'] = header_table['Voucher No.'].astype(str).str.strip().str.upper()
                if 'Voucher No.' in data_df.columns:
                    data_df.loc[:, 'Voucher No.'] = data_df['Voucher No.'].astype(str).str.strip().str.upper()

                if 'Part Supply' in data_df.columns:
                    data_df.loc[:, 'Part Supply'] = data_df['Part Supply'].ffill()

                    def get_part_supply(vno):
                        vals = data_df[data_df['Voucher No.'] == vno]['Part Supply'].dropna().unique()
                        vals = [v for v in vals if v in ("OE", "SPD", "OE SPD")]
                        return ', '.join(sorted(set(vals))) if len(vals) > 0 else ''

                    header_table['Part Supply'] = header_table['Voucher No.'].apply(get_part_supply)

            self.current_df = header_table

            if 'Date' in header_table.columns and not header_table['Date'].isnull().all():
                min_date = pd.to_datetime(header_table['Date'], errors='coerce').min()
                max_date = pd.to_datetime(header_table['Date'], errors='coerce').max()
                if pd.notnull(min_date):
                    min_qdate = QDate(min_date.year, min_date.month, min_date.day)
                    self.from_date.setMinimumDate(min_qdate)
                    self.to_date.setMinimumDate(min_qdate)
                    self.from_date.setDate(min_qdate)
                if pd.notnull(max_date):
                    max_qdate = QDate(max_date.year, max_date.month, max_date.day)
                    self.from_date.setMaximumDate(max_qdate)
                    self.to_date.setMaximumDate(max_qdate)
                    self.to_date.setDate(max_qdate)

            self.apply_filters()
        else:
            self.table.setModel(PandasModel(pd.DataFrame()))
            self.current_df = pd.DataFrame()

    def apply_filters(self):
        df = self.current_df.copy()

        # Date filter
        if 'Date' in df.columns:
            from_date = self.from_date.date().toPyDate()
            to_date = self.to_date.date().toPyDate()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[(df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))]

        # Part Supply filter
        if 'Part Supply' in df.columns:
            if self.oe_radio.isChecked():
                df = df[df['Part Supply'].str.contains('OE', na=False)]
            elif self.spd_radio.isChecked():
                df = df[df['Part Supply'].str.contains('SPD', na=False)]
            elif self.oe_spd_radio.isChecked():
                df = df[df['Part Supply'].str.contains('OE SPD', na=False)]

        model = PandasModel(df)
        self.table.setModel(model)
        self.auto_fit_columns()

    def export_excel(self):
        if not self.current_df.empty:
            df = self.current_df.copy()
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save Excel File', '', 'Excel Files (*.xlsx)')
            if file_name:
                df.to_excel(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Invoice headers exported to Excel successfully!')

    def export_csv(self):
        if not self.current_df.empty:
            df = self.current_df.copy()
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save CSV File', '', 'CSV Files (*.csv)')
            if file_name:
                df.to_csv(file_name, index=False)
                QMessageBox.information(self, 'Export Successful', 'Invoice headers exported to CSV successfully!')

class MasterDataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(12)

        title_label = QLabel('🔧 Manage master data records (Part Numbers, Model, Category, Supply, PO Number, Color)')
        title_label.setStyleSheet("background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 10px; padding: 15px; border: 1px solid #90caf9; margin-top: 10px;")
        self.layout.addWidget(title_label)

        self.table = QTableWidget()
        self.table.setColumnCount(6)  # Updated for PO Number and Color
        self.table.setHorizontalHeaderLabels(["Part Number", "Model", "Part Category", "Part Supply", "PO Number", "Color"])
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: #f8fafc; background-color: #ffffff; font-size: 14px; border-radius: 8px;")
        self.layout.addWidget(self.table, stretch=1)

        input_group_box = QGroupBox("Record Details")
        input_group_box.setStyleSheet("QGroupBox { border: 1px solid #cbd5e1; border-radius: 8px; margin-top: 10px; font-size: 13px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        form_layout = QFormLayout()
        form_layout.setContentsMargins(10, 15, 10, 10)
        form_layout.setSpacing(8)

        input_style = "font-size: 13px; min-height: 22px; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 6px;"
        label_style = "font-size: 13px; color: #374151;"

        self.part_number_input = QLineEdit()
        self.part_number_input.setStyleSheet(input_style)
        self.model_input = QLineEdit()
        self.model_input.setStyleSheet(input_style)
        self.category_input = QLineEdit()
        self.category_input.setStyleSheet(input_style)
        self.supply_input = QComboBox()
        self.supply_input.addItems(["OE", "SPD", "OE SPD"])
        self.supply_input.setStyleSheet(input_style)
        self.po_number_input = QLineEdit()  # New field
        self.po_number_input.setStyleSheet(input_style)
        self.color_input = QLineEdit()      # New field
        self.color_input.setStyleSheet(input_style)

        form_layout.addRow(QLabel("Part Number:", styleSheet=label_style), self.part_number_input)
        form_layout.addRow(QLabel("Model:", styleSheet=label_style), self.model_input)
        form_layout.addRow(QLabel("Part Category:", styleSheet=label_style), self.category_input)
        form_layout.addRow(QLabel("Part Supply:", styleSheet=label_style), self.supply_input)
        form_layout.addRow(QLabel("PO Number:", styleSheet=label_style), self.po_number_input)
        form_layout.addRow(QLabel("Color:", styleSheet=label_style), self.color_input)

        input_group_box.setLayout(form_layout)
        self.layout.addWidget(input_group_box)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.setContentsMargins(0, 10, 0, 0)

        self.add_btn = QPushButton("Add/Update")
        self.delete_btn = QPushButton("Delete")
        self.import_btn = QPushButton("Import from Excel/CSV")
        self.export_btn = QPushButton("Export to Excel/CSV")

        button_style = "font-size: 14px; padding: 8px 16px; border-radius: 6px;"
        self.add_btn.setStyleSheet(button_style + "background: #1e40af; color: #fff;")
        self.delete_btn.setStyleSheet(button_style + "background: #dc3545; color: #fff;")
        self.import_btn.setStyleSheet(button_style + "background: #007bff; color: #fff;")
        self.export_btn.setStyleSheet(button_style + "background: #17a2b8; color: #fff;")

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.export_btn)
        self.layout.addLayout(btn_layout)

        self.setLayout(self.layout)

        self.add_btn.clicked.connect(self.add_or_update_record)
        self.delete_btn.clicked.connect(self.delete_record)
        self.import_btn.clicked.connect(self.import_data)
        self.export_btn.clicked.connect(self.export_data)
        self.table.cellClicked.connect(self.fill_inputs_from_table)

        self.load_data()

    def load_data(self):
        self.table.setRowCount(0)
        records = fetch_all_master_records()
        for row_idx, row in enumerate(records):
            for col_idx, value in enumerate(row[:6]):  # Updated to include PO and Color
                if self.table.rowCount() <= row_idx:
                    self.table.insertRow(row_idx)
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def add_or_update_record(self):
        part_number = self.part_number_input.text().strip()
        model = self.model_input.text().strip()
        category = self.category_input.text().strip()
        supply = self.supply_input.currentText()
        po_number = self.po_number_input.text().strip()  # New field
        color = self.color_input.text().strip()          # New field

        if not part_number:
            QMessageBox.warning(self, "Input Error", "Part Number is required.")
            return

        add_master_record(part_number, model, category, supply, po_number, color)
        self.load_data()
        self.clear_inputs()

    def delete_record(self):
        part_number = self.part_number_input.text().strip()
        if not part_number:
            QMessageBox.warning(self, "Input Error", "Select a record to delete.")
            return

        delete_master_record(part_number)
        self.load_data()
        self.clear_inputs()

    def fill_inputs_from_table(self, row, col):
        self.part_number_input.setText(self.table.item(row, 0).text())
        self.model_input.setText(self.table.item(row, 1).text())
        self.category_input.setText(self.table.item(row, 2).text())
        self.supply_input.setCurrentText(self.table.item(row, 3).text())
        self.po_number_input.setText(self.table.item(row, 4).text())  # New field
        self.color_input.setText(self.table.item(row, 5).text())      # New field

    def clear_inputs(self):
        self.part_number_input.clear()
        self.model_input.clear()
        self.category_input.clear()
        self.supply_input.setCurrentIndex(0)
        self.po_number_input.clear()  # New field
        self.color_input.clear()      # New field

    def import_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Import Master Data', '', 'Excel/CSV Files (*.xlsx *.xls *.csv)')
        if file_name:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_name, dtype=str)
            else:
                df = pd.read_excel(file_name, dtype=str)

            imported = 0

            def get_col(row, *names):
                for name in names:
                    if name in row and pd.notna(row[name]):
                        return str(row[name]).strip()
                return ''

            for _, row in df.iterrows():
                # Match exact column names from your Excel file
                part_number = get_col(row, 'PartNumber', 'Part Number', 'Part Numbers')
                model = get_col(row, 'Model')
                category = get_col(row, 'Part Description', 'Part  Description', 'Part Category', 'Description')
                supply = get_col(row, 'Part Supply ', 'Part Supply', 'PartSupply') or 'OE'
                po_number = get_col(row, 'PO Number', 'PONumber', 'PO_Number')  # New field
                color = get_col(row, 'Color ', 'Color', 'Colour')               # New field

                if part_number and part_number != 'nan' and len(part_number) > 0:
                    add_master_record(part_number, model, category, supply, po_number, color)
                    imported += 1

            self.load_data()
            QMessageBox.information(self, 'Import Successful', f'{imported} master data records imported successfully!')

    def export_data(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Export Master Data', '', 'Excel Files (*.xlsx);;CSV Files (*.csv)')
        if file_name:
            records = fetch_all_master_records()
            df = pd.DataFrame(records, columns=["Part Number", "Model", "Part Category", "Part Supply", "PO Number", "Color", "Normalized Part"])
            df = df[["Part Number", "Model", "Part Category", "Part Supply", "PO Number", "Color"]]

            if file_name.endswith('.csv'):
                df.to_csv(file_name, index=False)
            else:
                df.to_excel(file_name, index=False)

            QMessageBox.information(self, 'Export Successful', 'Master data exported successfully!')

class VisualizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)
        
        # Title
        title_label = QLabel('📊 Sales Data Visualizations & Analytics')
        title_label.setStyleSheet("background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 10px; padding: 15px; border: 1px solid #90caf9;")
        self.layout.addWidget(title_label)
        
        # Simplified Controls Layout - Single Row
        controls_layout = QHBoxLayout()
        
        # Chart type selection
        chart_group = QGroupBox("Chart Selection")
        chart_group.setMaximumWidth(300)
        chart_layout = QVBoxLayout()
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Sales by Part Supply", "Monthly Sales Trend", "Top 10 Customers by Value",
            "Sales by Model", "Quantity vs Value", "Daily Sales Volume",
            "Customer Distribution", "Part Category Analysis", "Color Analysis",
            "PO Number Analysis", "Sales by Destination", "Truck Trip Analysis"
        ])
        
        chart_layout.addWidget(QLabel("Chart Type:"))
        chart_layout.addWidget(self.chart_type_combo)
        chart_group.setLayout(chart_layout)
        controls_layout.addWidget(chart_group)
        
        # Date range selection
        date_group = QGroupBox("Date Range")
        date_group.setMaximumWidth(300)
        date_layout = QVBoxLayout()
        
        date_row_layout = QHBoxLayout()
        self.date_from = QDateEdit()
        self.date_to = QDateEdit()
        self.date_from.setCalendarPopup(True)
        self.date_to.setCalendarPopup(True)
        self.date_from.setDisplayFormat('yyyy-MM-dd')
        self.date_to.setDisplayFormat('yyyy-MM-dd')
        
        date_row_layout.addWidget(QLabel("From:"))
        date_row_layout.addWidget(self.date_from)
        date_row_layout.addWidget(QLabel("To:"))
        date_row_layout.addWidget(self.date_to)
        
        date_layout.addLayout(date_row_layout)
        date_group.setLayout(date_layout)
        controls_layout.addWidget(date_group)
        
        # Part Supply filter with radio buttons - more compact
        part_supply_group = QGroupBox("Part Supply Filter")
        part_supply_group.setMaximumWidth(300)
        part_supply_layout = QVBoxLayout()
        
        # Radio buttons in 2 rows
        radio_row1 = QHBoxLayout()
        radio_row2 = QHBoxLayout()
        
        self.all_parts_radio = QRadioButton('All')
        self.oe_parts_radio = QRadioButton('OE')
        self.spd_parts_radio = QRadioButton('SPD')
        self.oe_spd_parts_radio = QRadioButton('OE SPD')
        self.all_parts_radio.setChecked(True)
        
        radio_row1.addWidget(self.all_parts_radio)
        radio_row1.addWidget(self.oe_parts_radio)
        radio_row2.addWidget(self.spd_parts_radio)
        radio_row2.addWidget(self.oe_spd_parts_radio)
        
        part_supply_layout.addLayout(radio_row1)
        part_supply_layout.addLayout(radio_row2)
        part_supply_group.setLayout(part_supply_layout)
        controls_layout.addWidget(part_supply_group)
        
        # Generate button
        generate_group = QGroupBox("Generate")
        generate_group.setMaximumWidth(200)
        generate_layout = QVBoxLayout()
        
        self.generate_chart_btn = QPushButton('Generate Chart')
        self.generate_chart_btn.setStyleSheet("background: #1e40af; color: white; padding: 12px 20px; border-radius: 8px; font-size: 15px; font-weight: bold;")
        
        generate_layout.addWidget(self.generate_chart_btn)
        generate_group.setLayout(generate_layout)
        controls_layout.addWidget(generate_group)
        
        controls_layout.addStretch()
        
        # Add controls to main layout
        self.layout.addLayout(controls_layout)
        
        # Chart display area with scroll - optimized size
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: 2px solid #e2e8f0; border-radius: 8px; background: #ffffff;")
        
        self.chart_widget = QLabel("Select a chart type and click 'Generate Chart' to view visualizations")
        self.chart_widget.setStyleSheet("border: 2px dashed #cbd5e1; border-radius: 8px; padding: 30px; text-align: center; color: #6b7280; font-size: 16px;")
        self.chart_widget.setAlignment(Qt.AlignCenter)
        
        scroll_area.setWidget(self.chart_widget)
        self.layout.addWidget(scroll_area, stretch=1)
        
        # Connect signals
        self.generate_chart_btn.clicked.connect(self.generate_chart)
        
        self.setLayout(self.layout)
        self.current_df = pd.DataFrame()
    
    def update_data(self, df):
        self.current_df = df.copy()
        if not df.empty and 'Date' in df.columns:
            min_date = pd.to_datetime(df['Date'], errors='coerce').min()
            max_date = pd.to_datetime(df['Date'], errors='coerce').max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                self.date_from.setDate(QDate(min_date.year, min_date.month, min_date.day))
                self.date_to.setDate(QDate(max_date.year, max_date.month, max_date.day))
    
    def apply_filters_to_data(self, df):
        # Apply date filter
        if 'Date' in df.columns:
            from_date = self.date_from.date().toPyDate()
            to_date = self.date_to.date().toPyDate()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[(df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))]
        
        # Apply part supply filter
        if 'Part Supply' in df.columns:
            if self.oe_parts_radio.isChecked():
                df = df[df['Part Supply'] == 'OE']
            elif self.spd_parts_radio.isChecked():
                df = df[df['Part Supply'] == 'SPD']
            elif self.oe_spd_parts_radio.isChecked():
                df = df[df['Part Supply'] == 'OE SPD']
        
        return df
    
    def generate_chart(self):
        if self.current_df.empty:
            QMessageBox.warning(self, "No Data", "Please load data first from the Data View tab.")
            return
            
        chart_type = self.chart_type_combo.currentText()
        
        # Apply all filters to data
        df = self.apply_filters_to_data(self.current_df.copy())
        
        if df.empty:
            QMessageBox.warning(self, "No Data", "No data available after applying filters.")
            return
        
        try:
            if chart_type == "Sales by Part Supply":
                self.create_part_supply_chart(df)
            elif chart_type == "Monthly Sales Trend":
                self.create_monthly_trend_chart(df)
            elif chart_type == "Top 10 Customers by Value":
                self.create_top_customers_chart(df)
            elif chart_type == "Sales by Model":
                self.create_model_sales_chart(df)
            elif chart_type == "Quantity vs Value":
                self.create_quantity_value_chart(df)
            elif chart_type == "Daily Sales Volume":
                self.create_daily_volume_chart(df)
            elif chart_type == "Customer Distribution":
                self.create_customer_distribution_chart(df)
            elif chart_type == "Part Category Analysis":
                self.create_part_category_chart(df)
            elif chart_type == "Color Analysis":
                self.create_color_analysis_chart(df)
            elif chart_type == "PO Number Analysis":
                self.create_po_analysis_chart(df)
            elif chart_type == "Sales by Destination":
                self.create_destination_chart(df)
            elif chart_type == "Truck Trip Analysis":
                self.create_truck_trip_analysis_chart(df)
                
        except Exception as e:
            QMessageBox.warning(self, "Chart Error", f"Error generating chart: {str(e)}")
    
    def create_truck_trip_analysis_chart(self, df):
        if 'Despatch Through' not in df.columns or 'Date' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Despatch Through or Date column not found.")
            return
        
        # Convert Date to datetime for proper grouping
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Group by truck and date (one trip per truck per day)
        truck_trips = df.groupby(['Despatch Through', df['Date'].dt.date]).size().reset_index(name='trip_count')
        truck_trips['trip_count'] = 1  # Each group represents one trip
        
        # Count total trips per truck
        trips_per_truck = truck_trips.groupby('Despatch Through')['trip_count'].sum().reset_index()
        trips_per_truck = trips_per_truck.sort_values('trip_count', ascending=True).tail(15)  # Top 15 trucks
        
        plt.figure(figsize=(12, 8))  # Reduced size for better screen fit
        bars = plt.barh(trips_per_truck['Despatch Through'], trips_per_truck['trip_count'], 
                       color='#1e40af', alpha=0.8)
        
        plt.title('Number of Trips by Truck (One trip per day per truck)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Number of Trips', fontsize=12)
        plt.ylabel('Truck / Despatch Through', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} trips', ha='left', va='center', fontweight='bold', fontsize=10)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_part_supply_chart(self, df):
        if 'Part Supply' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Part Supply or Gross Total column not found.")
            return
            
        supply_sales = df.groupby('Part Supply')['Gross Total'].sum().reset_index()
        supply_sales = supply_sales.sort_values('Gross Total', ascending=False)
        
        plt.figure(figsize=(10, 6))  # Optimized size
        colors = ['#1e40af', '#dc2626', '#059669', '#d97706']
        bars = plt.bar(supply_sales['Part Supply'], supply_sales['Gross Total'], color=colors[:len(supply_sales)])
        
        plt.title('Sales by Part Supply Type', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Part Supply Type', fontsize=12)
        plt.ylabel('Gross Total (₹)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_monthly_trend_chart(self, df):
        if 'Date' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Date or Gross Total column not found.")
            return
            
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_sales = df.groupby('Month')['Gross Total'].sum().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)
        
        plt.figure(figsize=(12, 6))  # Optimized size
        plt.plot(monthly_sales['Month'], monthly_sales['Gross Total'], 
                marker='o', linewidth=3, markersize=8, color='#1e40af')
        plt.fill_between(monthly_sales['Month'], monthly_sales['Gross Total'], alpha=0.3, color='#1e40af')
        
        plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Gross Total (₹)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_top_customers_chart(self, df):
        if 'Customer' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Customer or Gross Total column not found.")
            return
            
        customer_sales = df.groupby('Customer')['Gross Total'].sum().reset_index()
        customer_sales = customer_sales.sort_values('Gross Total', ascending=True).tail(10)
        
        plt.figure(figsize=(10, 8))  # Optimized size
        bars = plt.barh(customer_sales['Customer'], customer_sales['Gross Total'], color='#2563eb')
        
        plt.title('Top 10 Customers by Sales Value', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Gross Total (₹)', fontsize=12)
        plt.ylabel('Customer', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'₹{width:,.0f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_model_sales_chart(self, df):
        if 'Model' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Model or Gross Total column not found.")
            return
            
        model_sales = df.groupby('Model')['Gross Total'].sum().reset_index()
        model_sales = model_sales[model_sales['Model'] != ''].sort_values('Gross Total', ascending=False)
        
        plt.figure(figsize=(10, 8))  # Optimized size
        colors = plt.cm.Set3(range(len(model_sales)))
        wedges, texts, autotexts = plt.pie(model_sales['Gross Total'], labels=model_sales['Model'], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        plt.title('Sales Distribution by Model', fontsize=16, fontweight='bold', pad=15)
        plt.axis('equal')
        
        self.save_and_display_chart()
    
    def create_quantity_value_chart(self, df):
        if 'Quantity' not in df.columns or 'Value' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Quantity or Value column not found.")
            return
        
        # Remove null values
        df_clean = df.dropna(subset=['Quantity', 'Value'])
        
        plt.figure(figsize=(10, 6))  # Optimized size
        plt.scatter(df_clean['Quantity'], df_clean['Value'], alpha=0.6, color='#1e40af', s=50)
        
        plt.title('Quantity vs Value Analysis', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Quantity', fontsize=12)
        plt.ylabel('Value (₹)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_color_analysis_chart(self, df):
        if 'Color' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Color or Gross Total column not found.")
            return
            
        color_sales = df.groupby('Color')['Gross Total'].sum().reset_index()
        color_sales = color_sales[color_sales['Color'] != ''].sort_values('Gross Total', ascending=False).head(12)
        
        plt.figure(figsize=(12, 6))  # Optimized size
        bars = plt.bar(color_sales['Color'], color_sales['Gross Total'], color='#059669')
        
        plt.title('Sales by Color (Top 12)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Color', fontsize=12)
        plt.ylabel('Gross Total (₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=45, fontsize=8)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_po_analysis_chart(self, df):
        if 'PO Number' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "PO Number or Gross Total column not found.")
            return
            
        po_sales = df.groupby('PO Number')['Gross Total'].sum().reset_index()
        po_sales = po_sales[po_sales['PO Number'] != ''].sort_values('Gross Total', ascending=False).head(10)
        
        plt.figure(figsize=(10, 8))  # Optimized size
        bars = plt.barh(po_sales['PO Number'], po_sales['Gross Total'], color='#d97706')
        
        plt.title('Top 10 PO Numbers by Sales Value', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Gross Total (₹)', fontsize=12)
        plt.ylabel('PO Number', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'₹{width:,.0f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_destination_chart(self, df):
        if 'Destination' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Destination or Gross Total column not found.")
            return
            
        dest_sales = df.groupby('Destination')['Gross Total'].sum().reset_index()
        dest_sales = dest_sales.sort_values('Gross Total', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))  # Optimized size
        colors = plt.cm.viridis(range(len(dest_sales)))
        bars = plt.bar(dest_sales['Destination'], dest_sales['Gross Total'], color=colors)
        
        plt.title('Sales by Destination (Top 10)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Destination', fontsize=12)
        plt.ylabel('Gross Total (₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=45, fontsize=8)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_daily_volume_chart(self, df):
        if 'Date' not in df.columns or 'Quantity' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Date or Quantity column not found.")
            return
            
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        daily_qty = df.groupby(df['Date'].dt.date)['Quantity'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))  # Optimized size
        plt.plot(daily_qty['Date'], daily_qty['Quantity'], 
                marker='o', linewidth=2, markersize=5, color='#dc2626')
        plt.fill_between(daily_qty['Date'], daily_qty['Quantity'], alpha=0.3, color='#dc2626')
        
        plt.title('Daily Sales Volume', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Quantity', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def create_customer_distribution_chart(self, df):
        if 'Customer' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Customer column not found.")
            return
            
        customer_count = df['Customer'].value_counts().head(8)
        
        plt.figure(figsize=(8, 8))  # Optimized size
        colors = plt.cm.Pastel1(range(len(customer_count)))
        wedges, texts, autotexts = plt.pie(customer_count.values, labels=customer_count.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        # Create donut chart
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title('Customer Distribution (Top 8)', fontsize=16, fontweight='bold', pad=15)
        plt.axis('equal')
        
        self.save_and_display_chart()
    
    def create_part_category_chart(self, df):
        if 'Part Category' not in df.columns or 'Gross Total' not in df.columns:
            QMessageBox.warning(self, "Missing Data", "Part Category or Gross Total column not found.")
            return
            
        category_sales = df.groupby('Part Category')['Gross Total'].sum().reset_index()
        category_sales = category_sales[category_sales['Part Category'] != ''].sort_values('Gross Total', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))  # Optimized size
        bars = plt.bar(category_sales['Part Category'], category_sales['Gross Total'], color='#7c3aed')
        
        plt.title('Sales by Part Category (Top 10)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Part Category', fontsize=12)
        plt.ylabel('Gross Total (₹)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'₹{height:,.0f}', ha='center', va='bottom', fontweight='bold', rotation=45, fontsize=8)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        self.save_and_display_chart()
    
    def save_and_display_chart(self):
        # Save chart to temporary file
        chart_path = 'temp_chart.png'
        plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='white')  # Reduced DPI for better performance
        plt.close()
        
        # Load and display chart - optimized for screen fit
        pixmap = QPixmap(chart_path)
        scaled_pixmap = pixmap.scaled(1000, 650, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Better screen fit
        self.chart_widget.setPixmap(scaled_pixmap)
        
        # Clean up temp file
        if os.path.exists(chart_path):
            os.remove(chart_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Honda Sales Report Visualizer - Optimized Screen Fit')
        self.resize(1400, 900)  # Standard screen fit
        self.setWindowIcon(QIcon('app.ico'))
        self.setFont(QFont('Segoe UI', 10))  # Slightly smaller font for better fit

        self.tabs = QTabWidget()

        self.data_tab = DataTab()
        self.part_tab = PartTab()
        self.master_tab = MasterDataTab()
        self.header_tab = HeaderTab()
        self.visualization_tab = VisualizationTab()  # Optimized visualization tab

        self.tabs.addTab(self.data_tab, 'Data View')
        self.tabs.addTab(self.part_tab, 'Part Numbers')
        self.tabs.addTab(self.master_tab, 'Master Data')
        self.tabs.addTab(self.header_tab, 'Invoice Headers')
        self.tabs.addTab(self.visualization_tab, '📊 Charts & Analytics')

        self.setCentralWidget(self.tabs)

        # Link tabs for communication
        self.data_tab.parent_tabs = self.tabs
        self.part_tab.parent_tabs = self.tabs
        self.header_tab.parent_tabs = self.tabs
        self.visualization_tab.parent_tabs = self.tabs

        # Set tab references for easy access
        self.tabs.data_tab = self.data_tab
        self.tabs.part_tab = self.part_tab
        self.tabs.header_tab = self.header_tab
        self.tabs.master_tab = self.master_tab
        self.tabs.visualization_tab = self.visualization_tab

if __name__ == '__main__':
    init_master_db()
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    app.setStyleSheet(APP_STYLESHEET)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
