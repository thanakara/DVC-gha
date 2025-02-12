import os
import xlrd
import numpy as np
import matplotlib.pyplot as plt
from linear_model import LinearRegression


def get_data():
    def _test_path():
        DATA_FILE = os.path.join(os.path.dirname(__file__), "fire_theft.xls")
        try:
            assert os.path.exists(DATA_FILE) is True
            return DATA_FILE
        except AssertionError:
            raise FileNotFoundError
        
    book = xlrd.open_workbook(_test_path(), encoding_override="utf-8")
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    X = data.T[0].reshape(-1, 1)
    Y = data.T[1].reshape(-1, 1)
    return X, Y

def main():
    X, Y = get_data()
    model = LinearRegression("lin_reg")
    model.fit(X, Y)

    fig = plt.figure(figsize=(8, 4))
    fig.set_facecolor("lightgray")
    plt.scatter(X, Y, c="orange", edgecolors="k", label="Ground Truth")
    plt.plot(X, model(X), "k-", label="Predicted Data")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()


if __name__ == "__main__":
    main()
