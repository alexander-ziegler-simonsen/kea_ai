import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn import datasets # maybe not needed
    import matplotlib.pyplot as plt
    import time
    return datasets, np, pd, plt, time


@app.cell
def _(pd):
    cars = pd.read_csv("./cars.csv")

    cars.head()
    return (cars,)


if __name__ == "__main__":
    app.run()
