"""Quick script to analyze Task 2 dataset structure"""
import pandas as pd
import json
import os

os.chdir(r"c:\Users\HP\OneDrive\Desktop\TASK1\task2\dataset")

# 1. Check master_clauses.csv
print("="*60)
print("1. MASTER_CLAUSES.CSV ANALYSIS")
print("="*60)
df = pd.read_csv("master_clauses.csv")
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}):")
for col in df.columns[:20]:
    print(f"  - {col}")
print("  ... and more")

# 2. Check CUAD_v1.json
print("\n" + "="*60)
print("2. CUAD_V1.JSON ANALYSIS")
print("="*60)
with open("CUAD_v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Total contracts: {len(data['data'])}")
print(f"Sample contract: {data['data'][0]['title']}")
print(f"Number of QA pairs in first contract: {len(data['data'][0]['paragraphs'][0]['qas'])}")

# 3. Check Excel files
print("\n" + "="*60)
print("3. EXCEL FILES ANALYSIS")
print("="*60)
excel_dir = "label_group_xlsx"
excel_files = os.listdir(excel_dir)
print(f"Total Excel files: {len(excel_files)}")

# Sample one excel
df_excel = pd.read_excel(f"{excel_dir}/Label Report - Governing Law.xlsx")
print(f"\nSample Excel (Governing Law):")
print(f"  Shape: {df_excel.shape}")
print(f"  Columns: {df_excel.columns.tolist()}")
print(f"  First 3 rows:")
print(df_excel.head(3))
