import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from sklearn import datasets
    import matplotlib.pyplot as plt
    return datasets, np, plt


if __name__ == "__main__":
    app.run()
