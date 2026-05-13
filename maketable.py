import xlsxwriter

# Define file path
excel_path = "KPMS_Avg_Daily_Entry_Exit_Details.xlsx"

# Create an Excel workbook and worksheet
workbook = xlsxwriter.Workbook(excel_path)
worksheet = workbook.add_worksheet("KPMS Details")

# Define formats
header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})

# Titles
worksheet.merge_range("A1:R1", "Avg daily Hourly Entry Details of KPMS (one Week Avg)", header_format)
worksheet.write("A2", "Sr. No", header_format)
worksheet.write("B2", "Date", header_format)
entry_hours = [
    "0600 Hrs to 0700 hrs", "0700 Hrs to 0800 hrs", "0800 Hrs to 0900 hrs", "0900 Hrs to 1000 hrs",
    "1000 Hrs to 1100 hrs", "1100 Hrs to 1200 hrs", "1200 Hrs to 1300 hrs", "1300 Hrs to 1400 hrs",
    "1400 Hrs to 1500 hrs", "1500 Hrs to 1600 hrs", "1600 Hrs to 1700 hrs", "1700 Hrs to 1800 hrs",
    "1800 Hrs to 1900 hrs", "1900 Hrs to 2000 hrs", "2000 Hrs to 2100 hrs", "2100 Hrs to 2200 hrs",
    "2200 Hrs to 2300 hrs", "2300 Hrs to 0000 hrs"
]
entry_values = [24, 101, 242, 186, 357, 102, 34, 45, 61, 27, 20, 106, 59, 52, 78, 106, 4, 0]

for col, hour in enumerate(entry_hours, start=2):
    worksheet.write(1, col, hour, header_format)
worksheet.write_row("A3", [1, "KPMS"] + entry_values, cell_format)

# Offset for exit details
start_row_exit = 5
worksheet.merge_range(start_row_exit, 0, start_row_exit, 17, "Avg daily Hourly Exit Details of KPMS (one Week Avg)", header_format)
worksheet.write(start_row_exit+1, 0, "Sr. No", header_format)
worksheet.write(start_row_exit+1, 1, "Date", header_format)

exit_values = [28, 76, 54, 63, 81, 61, 64, 83, 90, 77, 99, 169, 306, 277, 162, 91, 38, 0]
for col, hour in enumerate(entry_hours, start=2):
    worksheet.write(start_row_exit+1, col, hour, header_format)
worksheet.write_row(start_row_exit+2, 0, [1, "KPMS"] + exit_values, cell_format)

# Summary section
start_row_summary = start_row_exit + 5
summary_data = [
    ["Avg daily NCMC Entry Details of KPMS (one Week Avg)", 1284],
    ["Avg daily NCMC Exit Details of KPMS (one Week Avg)", 1289],
    ["Avg daily QR Entry Details of KPMS (one Week Avg)", 4849],
    ["Avg daily QR Exit Details of KPMS (one Week Avg)", 4357],
]
worksheet.write(start_row_summary, 0, "Description", header_format)
worksheet.write(start_row_summary, 1, "Qty", header_format)
for i, row in enumerate(summary_data):
    worksheet.write(start_row_summary + i + 1, 0, row[0], cell_format)
    worksheet.write(start_row_summary + i + 1, 1, row[1], cell_format)

workbook.close()
print(f"Excel file saved to: {excel_path}")
