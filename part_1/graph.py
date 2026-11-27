import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from tkinter import messagebox

CSV_PATH = "resultst.csv"
COLUMNS = ["date", "personnes"]


def load_data(path=CSV_PATH, day=None, start=None, end=None):
    """
    Charge et nettoie les données.
    day   : "2026-11-16"  (optionnel)
    start : "20:00"       (optionnel, début de la plage horaire)
    end   : "21:00"       (optionnel, fin de la plage horaire)
    """
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Conversion robuste : une ligne corrompue (ou un éventuel en-tête) devient NaN
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["personnes"] = pd.to_numeric(df["personnes"], errors="coerce")
    df = df.dropna().sort_values("date")
    df["personnes"] = df["personnes"].astype(int)

    if day:
        df = df[df["date"].dt.date == pd.to_datetime(day).date()]

    if start and end:
        df = df.set_index("date").between_time(start, end).reset_index()

    return df


def show_graph(day=None, start=None, end=None, smooth_window=5):
    try:
        df = load_data(day=day, start=start, end=end)

        if df.empty:
            messagebox.showwarning("Attention", "Aucune donnée à afficher.")
            return

        fig, ax = plt.subplots(figsize=(12, 5.5))

        # Courbe en escalier : le nombre de personnes reste constant entre deux mesures
        ax.step(df["date"], df["personnes"], where="post",
                color="#1f77b4", linewidth=1.8, label="Détections")
        ax.fill_between(df["date"], df["personnes"], step="post",
                        color="#1f77b4", alpha=0.12)

        # Moyenne mobile pour lisser le bruit de détection
        if len(df) >= smooth_window:
            trend = df["personnes"].rolling(smooth_window, center=True,
                                            min_periods=1).mean()
            ax.plot(df["date"], trend, color="#ff7f0e", linewidth=2,
                    linestyle="--", label=f"Tendance (moy. {smooth_window} mesures)")

        # Moyenne globale
        mean_val = df["personnes"].mean()
        ax.axhline(mean_val, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Moyenne : {mean_val:.1f}")

        # Annotation du pic
        peak = df.loc[df["personnes"].idxmax()]
        ax.annotate(
            f"Pic : {peak['personnes']} pers.\n{peak['date']:%H:%M:%S}",
            xy=(peak["date"], peak["personnes"]),
            xytext=(0, 25), textcoords="offset points",
            ha="center", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
        )

        # Axe X : format automatique selon l'étendue (heures, minutes, jours)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        # Axe Y : uniquement des entiers, démarre à 0
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, df["personnes"].max() + 2)

        # Titre dynamique
        t0, t1 = df["date"].min(), df["date"].max()
        ax.set_title(
            f"Fréquentation détectée — {t0:%d/%m/%Y %H:%M} → {t1:%H:%M}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Heure")
        ax.set_ylabel("Nombre de personnes")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        plt.show()

    except FileNotFoundError:
        messagebox.showerror("Erreur", f"Fichier introuvable : {CSV_PATH}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))