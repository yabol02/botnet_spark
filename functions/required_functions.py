from pyspark import RDD
import numpy as np
# ---------------- FUNCTIONS FOR EXECISE 1 -------
def readFile(filename: str, sc) -> RDD:
    rdd = sc.textFile(filename)
    return (
        rdd.map(lambda x: x.split(","))
        .map(lambda x: tuple(map(float, x)))
        .map(lambda x: (x[:-1], int(x[-1])))
    )

def normalize(rdd_xy: RDD) -> RDD:
    rdd_X = rdd_xy.map(lambda x: x[0])
    _ = rdd_xy.map(lambda x: x[1])  # <--- CUIDADO, no se usa, así que habría que borrarlo

    mu, rows = rdd_X.map(lambda x: (np.array(x), 1)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    mu /= rows
    std = (rdd_X.map(lambda x: (np.array(x) - mu) ** 2).reduce(lambda x, y: x + y) / rows) ** 0.5

    return rdd_xy.map(lambda x: ((np.array(x[0]) - mu) / std, x[1]))

def train(rdd_xy: RDD, iterations: int, learning_rate: float, lambda_reg: float, show_logs: bool = True) -> tuple[np.ndarray, float]:
    its_to_print = list(range(0, iterations, iterations // 10)) + [iterations]
    k = len(rdd_xy.first()[0])
    m = rdd_xy.count()
    w = np.random.rand(k)
    b = 0

    for it in range(iterations):
        # Gradients
        dw = (
            rdd_xy.map(lambda x: (predict_proba(w, b, x[0]) - x[1]) * np.array(x[0]))
            .reduce(lambda x, y: x + y) / m
        )
        db = (
            rdd_xy.map(lambda x: predict_proba(w, b, x[0]) - x[1])
            .reduce(lambda x, y: x + y) / m
        )

        # Regularization
        dw += lambda_reg / k * w

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Logging
        if show_logs and it in its_to_print:
            c = cost(w, b, rdd_xy, lambda_reg, k, m)
            acc = accuracy(w, b, rdd_xy)
            print(f"Iteration {it}/{iterations}, Cost: {c:.4f}, Accuracy: {acc:.4f}")

    return w, b


def sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(logit, -250, 250)))


def predict_proba(w: np.ndarray, b: float, X: np.ndarray) -> float:
    X = np.array(X)
    w = np.array(w)
    return sigmoid(np.dot(w, X) + b)


def predict(w: np.array, b: float, X: np.array) -> int:
    return 1 if predict_proba(w, b, X) > 0.5 else 0


def accuracy(w: np.ndarray, b: float, rdd_xy: RDD) -> float:
    preds = rdd_xy.map(lambda x: (int(predict(w, b, x[0]) == x[1]), 1)).reduce(
        lambda x, y: (x[0] + y[0], x[1] + y[1])
    )
    return preds[0] / preds[1]


def cost(w: np.ndarray, b: float, rdd_xy: RDD, lambda_reg: float, k: int, m: int) -> float:
    return (
        rdd_xy.map(lambda x: (predict_proba(w, b, x[0]), x[1]))
        .map(lambda x: (x[1] * np.log(x[0]) + (1 - x[1]) * np.log(1 - x[0])))
        .reduce(lambda x, y: x + y) / (-m) + lambda_reg * (w**2).sum() / (2 * k)
    )


# ------- FUNCTIONS FOR EXERCISE 2 -------

def transform(rdd_xy: RDD, blocks: int) -> RDD:
    return rdd_xy.map(lambda x: (x, np.random.randint(0, blocks))).cache()

def get_block_data(rdd_blocked: RDD, block_id: int) -> tuple[RDD, RDD]:
    train_rdd = rdd_blocked.flatMap(lambda x: [x[0]] if x[1] != block_id else [])
    test_rdd = rdd_blocked.flatMap(lambda x: [x[0]] if x[1] == block_id else [])
    return train_rdd, test_rdd