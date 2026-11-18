import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tkinter import messagebox

def show_graph():
    try:
        df = pd.read_csv("resultst.csv")

        if len(df) == 0:
            messagebox.showwarning(
                "Attention",
                "Aucune donnée à afficher."
            )
            return

        df["date"] = pd.to_datetime(df["date"])

        plt.figure(figsize=(10, 5))

        plt.plot(
            df["date"],
            df["personnes"],
            marker="o",
            color="blue"
        )

        # Afficher uniquement l'heure (HH:MM:SS) sur l'axe des abscisses
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=45)  # Évite que les heures ne se chevauchent

        plt.title("Nombre de personnes en fonction du temps")
        plt.xlabel("Temps")
        plt.ylabel("Nombre de personnes")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror(
            "Erreur",
            str(e)
        )