import numpy as np
from pyspark import RDD

# Notación en comentarios de transformaciones RDD:
#   X  → np.ndarray (vector de features)
#   y  → escalar (label o valor numérico)
#   b  → id de bloque (entero)
#   [] → elemento descartado (flatMap vacío)


# ---------------- FUNCTIONS FOR EXECISE 1 -------
def readFile(filename: str, sc) -> RDD:
    rdd = sc.textFile(filename)
    return (
        rdd.map(lambda x: x.split(","))             # "f0,f1,...,y" -> [f0, f1, ..., y]
            .map(lambda x: tuple(map(float, x)))    # [f0,..., y]   -> (f0,..., y)
            .map(lambda x: (x[:-1], int(x[-1])))    # (f0,..., y)   -> (X, y)
    )


def normalize(rdd_xy: RDD) -> RDD:
    rdd_X = rdd_xy.map(lambda x: x[0])  # (X, y) -> X

    sum_x, sum_x2, n = (
        rdd_X.map(lambda x: (np.array(x), np.array(x) ** 2, 1))             # X -> (X, X², 1)
            .reduce(lambda a, c: (a[0] + c[0], a[1] + c[1], a[2] + c[2]))   # (X, X², 1) -> (ΣX, ΣX², n)
    )
    mu = sum_x / n
    std = np.sqrt(sum_x2 / n - mu**2)

    return rdd_xy.map(lambda x: ((np.array(x[0]) - mu) / std, x[1]))    # (X, y) -> (X_norm, y)


def train(
    rdd_xy: RDD,
    iterations: int,
    learning_rate: float,
    lambda_reg: float,
    show_logs: bool = True,
) -> tuple[np.ndarray, float]:
    its_to_print = list(range(0, iterations, iterations // 10)) + [iterations]
    k = len(rdd_xy.first()[0])
    m = rdd_xy.count()
    w = np.random.rand(k)
    b = 0

    for it in range(iterations):
        # Gradients
        dw, db = (
            rdd_xy.map(lambda x: (lambda err: (err * np.array(x[0]), err))(predict_proba(w, b, x[0]) - x[1]))   # (X, y) -> (ε·X, ε)
                .reduce(lambda a, acc: (a[0] + acc[0], a[1] + acc[1]))                                          # (ε·X, ε) -> (Σε·X, Σε)
        )
        dw /= m
        db /= m

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
    return sigmoid(np.dot(np.array(w), np.array(X)) + b)


def predict(w: np.ndarray, b: float, X: np.ndarray) -> int:
    return 1 if predict_proba(w, b, X) > 0.5 else 0


def accuracy(w: np.ndarray, b: float, rdd_xy: RDD) -> float:
    correct, total = (
        rdd_xy.map(lambda x: (int(predict(w, b, x[0]) == x[1]), 1))     # (X, y) -> (acierto, 1)
            .reduce(lambda a, acc: (a[0] + acc[0], a[1] + acc[1]))      # (acierto, 1) -> (Σaciertos, n)
    )
    return correct / total


def cost(
    w: np.ndarray, b: float, rdd_xy: RDD, lambda_reg: float, k: int, m: int
) -> float:
    return (
        rdd_xy.map(lambda x: (lambda p, y: y * np.log(p) + (1 - y) * np.log(1 - p))(predict_proba(w, b, x[0]), x[1]))  # (X, y) -> y
            .reduce(lambda a, acc: a + acc) / (-m) + lambda_reg * (w**2).sum() / (2 * k)
    )


# ------- FUNCTIONS FOR EXERCISE 2 -------


def transform(rdd_xy: RDD, blocks: int) -> RDD:
    return rdd_xy.map(lambda x: (x, np.random.randint(0, blocks))).cache()  # (X, y) -> ((X, y), b)


def get_block_data(rdd_blocked: RDD, block_id: int) -> tuple[RDD, RDD]:
    train_rdd = rdd_blocked.flatMap(lambda x: [x[0]] if x[1] != block_id else [])   # ((X, y), b) -> (X, y) | []
    test_rdd = rdd_blocked.flatMap(lambda x: [x[0]] if x[1] == block_id else [])    # ((X, y), b) -> (X, y) | []
    return train_rdd, test_rdd
