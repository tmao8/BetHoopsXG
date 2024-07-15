import tkinter as tk
from tkinter import ttk
import predict
import pandas as pd
from datetime import datetime


class DataFrameDisplayApp:
    def __init__(self, master):
        self.master = master
        today_date = datetime.now().strftime("%m/%d")
        self.master.title("Points Predictions for " + today_date)
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width}x{screen_height}+0+0")

        data = predict.predict()
        self.df = pd.DataFrame(data)
        self.tree = ttk.Treeview(
            self.master, columns=list(self.df.columns), show="headings"
        )
        for column in self.df.columns:
            self.tree.heading(column, text=column)
        self.tree.pack(fill="both", expand=True)

        for index, row in self.df.iterrows():
            self.tree.insert("", "end", values=row.tolist())


def main():
    root = tk.Tk()
    app = DataFrameDisplayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
