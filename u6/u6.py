import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from sklearn import datasets
    import matplotlib.pyplot as plt
    import time
    return datasets, np, plt, time


@app.cell
def _(datasets, np):
    np.random.seed(42)

    m = 1_000
    split_train = int(m * 0.7)
    split_val = int(m * 0.15 + split_train)
    split_test = int(m * 0.15 + split_val)

    X, y = datasets.make_moons(
        n_samples=m, 
        noise=0.1, 
        random_state=0
    )

    X_train, y_train = X[:split_train], y[:split_train]
    X_val, y_val = X[split_train:split_val], y[split_train:split_val]
    X_test, y_test = X[split_val:split_test], y[split_val:split_test]
    return (
        X,
        X_test,
        X_train,
        X_val,
        m,
        split_test,
        split_train,
        split_val,
        y,
        y_test,
        y_train,
        y_val,
    )


@app.cell
def _(X_test, X_train, X_val, split_test, split_train, split_val):
    print('Splits:', split_train, split_val, split_test)
    print('Lenghths:', len(X_train), len(X_val), len(X_test))
    return


@app.cell
def _(X, plt, y):
    colors = ['blue' if label == 1 else 'red' for label in y]
    plt.scatter(X[:,0], X[:,1], color=colors)
    plt.show()
    y.shape, X.shape
    return (colors,)


@app.cell
def _(X_train, y_train):
    X_train[0], y_train[0]
    return


@app.cell
def _():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras import metrics
    return keras, layers, losses, metrics, models, optimizers, tf


@app.cell
def _(layers, models):
    model = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='sigmoid'),
        layers.Dense(10, activation='sigmoid'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model,)


@app.cell
def _(losses, metrics, model, optimizers):
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(model):
    model.summary()
    return


@app.cell
def _():
    epochs = 1_000
    return (epochs,)


@app.cell
def _(X_train, epochs, model, y_train):
    history = model.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=epochs, 
            shuffle=True)
    return (history,)


@app.cell
def _(X_train, model, y_train):
    model.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model, y_val):
    model.evaluate(X_val, y_val)
    return


@app.cell
def _(np, plt):
    def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
        cmap = plt.get_cmap(cmap)

        # Define region of interest by data limits
        xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
        ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
        steps = 1000
        x_span = np.linspace(xmin, xmax, steps)
        y_span = np.linspace(ymin, ymax, steps)
        xx, yy = np.meshgrid(x_span, y_span)

        # Make predictions across region of interest
        labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Plot decision boundary in region of interest
        z = labels.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

        # Get predicted labels on training data and plot
        train_labels = model.predict(X)
        ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)

        return fig, ax
    return (plot_decision_boundary,)


@app.cell
def _(X, model, plot_decision_boundary, y):
    #plot_decision_boundary(X, y, model, cmap='RdBu')
    plot_decision_boundary(X, y, model, cmap='RdBu')
    return


@app.cell
def _(epochs, history, plt):
    plt.plot([i for i in range(epochs)], history.history['loss'])
    return


@app.cell
def _():
    epochz= 3_000
    return (epochz,)


@app.cell
def _(X_train, epochz, model, y_train):
    history1 = model.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=epochz, 
            shuffle=True)
    return (history1,)


@app.cell
def _(X_train, model, y_train):
    model.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model, y_val):
    model.evaluate(X_val, y_val)
    return


@app.cell
def _(epochs, history, plt):
    plt.plot([i for i in range(epochs)], history.history['loss'])
    return


@app.cell
def _(X, model, plot_decision_boundary, y):
    plot_decision_boundary(X, y, model, cmap='RdBu')
    return


@app.cell
def _():
    # cant del in this editor
    return


@app.cell
def _(layers, models):
    model2 = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model2,)


@app.cell
def _(losses, metrics, model2, optimizers):
    model2.compile(
        optimizer=optimizers.SGD(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(X_train, model2, y_train):
    history2 = model2.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=1000, 
            shuffle=True
        )
    return (history2,)


@app.cell
def _(X_train, model2, y_train):
    model2.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model2, y_val):
    model2.evaluate(X_val, y_val)
    return


@app.cell
def _(epochs, history2, plt):
    plt.plot([i for i in range(epochs)], history2.history['loss'])
    return


@app.cell
def _(X, model2, plot_decision_boundary, y):
    plot_decision_boundary(X,y, model2, cmap='RdBu')
    return


@app.cell
def _():
    # vi bruger adam nu
    return


@app.cell
def _(layers, models):
    model3 = models.Sequential([
        layers.Flatten(input_shape=(2,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return (model3,)


@app.cell
def _(losses, metrics, model3, optimizers):
    model3.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), 
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )
    return


@app.cell
def _(X_train, model3, y_train):
    history3 = model3.fit(
            X_train, 
            y_train, 
            verbose=0, 
            epochs=1000, 
            shuffle=True
        )
    return (history3,)


@app.cell
def _(X_train, model, y_train):
    model.evaluate(X_train, y_train)
    return


@app.cell
def _(X_val, model, y_val):
    model.evaluate(X_val, y_val)
    return


@app.cell
def _(history3, plt):
    plt.plot([i for i in range(1000)], history3.history['loss'])
    return


@app.cell
def _(X, model3, plot_decision_boundary, y):
    plot_decision_boundary(X, y, model3, cmap='RdBu')
    return


@app.cell
def _():
    import random
    return (random,)


@app.cell
def _(random):
    random.seed(42)

    return


@app.cell
def _(random):
    random.randint(0,1000)
    return


if __name__ == "__main__":
    app.run()
