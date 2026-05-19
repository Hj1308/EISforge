import pandas as pd

# مسیر فایل idf خود را اینجا بگذارید
filepath = r"C:\Users\hoda\Desktop\YOUR_FILE.idf"

# امتحان با encoding های مختلف
for enc in ["latin-1", "cp1252", "utf-8"]:
    try:
        df = pd.read_csv(filepath, encoding=enc, sep=None,
                         engine="python", comment="#")
        print(f"Encoding: {enc}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(5))
        break
    except Exception as e:
        print(f"{enc} failed: {e}")