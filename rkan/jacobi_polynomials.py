def rational_jacobi_polynomial(x, n, alpha, beta, zeta, iota, backend):
    if n == 1:
        return (
            alpha - beta + (alpha + beta + 2) * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1)
        ) / 2
    elif n == 2:
        term1 = ((alpha + 1) * (alpha + 2)) / 2
        term2 = (
            (alpha + 2)
            * (3 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1)
        ) / 2
        term3 = (
            (3 + alpha + beta)
            * (4 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 2
        ) / 8
        return term1 + term2 + term3

    elif n == 3:
        term1 = ((alpha + 1) * (alpha + 2) * (3 + alpha)) / 6
        term2 = (
            (alpha + 2)
            * (3 + alpha)
            * (4 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1)
        ) / 4
        term3 = (
            (3 + alpha)
            * (4 + alpha + beta)
            * (5 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 2
        ) / 8
        term4 = (
            (4 + alpha + beta)
            * (5 + alpha + beta)
            * (6 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 3
        ) / 48
        return term1 + term2 + term3 + term4

    elif n == 4:
        term1 = ((alpha + 1) * (alpha + 2) * (3 + alpha) * (4 + alpha)) / 24
        term2 = (
            (alpha + 2)
            * (3 + alpha)
            * (4 + alpha)
            * (5 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1)
        ) / 12
        term3 = (
            (3 + alpha)
            * (4 + alpha)
            * (5 + alpha + beta)
            * (6 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 2
        ) / 16
        term4 = (
            (4 + alpha)
            * (5 + alpha + beta)
            * (6 + alpha + beta)
            * (7 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 3
        ) / 48
        term5 = (
            (5 + alpha + beta)
            * (6 + alpha + beta)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 4
        ) / 384
        return term1 + term2 + term3 + term4 + term5

    elif n == 5:
        term1 = (
            (alpha + 1) * (alpha + 2) * (3 + alpha) * (4 + alpha) * (5 + alpha)
        ) / 120
        term2 = (
            (alpha + 2)
            * (3 + alpha)
            * (4 + alpha)
            * (5 + alpha)
            * (6 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1)
        ) / 48
        term3 = (
            (3 + alpha)
            * (4 + alpha)
            * (5 + alpha)
            * (6 + alpha + beta)
            * (7 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 2
        ) / 48
        term4 = (
            (4 + alpha)
            * (5 + alpha)
            * (6 + alpha + beta)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 3
        ) / 96
        term5 = (
            (5 + alpha)
            * (6 + alpha + beta)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 4
        ) / 384
        term6 = (
            (6 + alpha + beta)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * (10 + alpha + beta)
            * (x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1) ** 5
        ) / 3840
        return term1 + term2 + term3 + term4 + term5 + term6

    elif n == 6:
        common_term = x**zeta / backend.sqrt(x ** (2 * zeta) + iota**2) - 1
        term1 = (
            (alpha + 1)
            * (alpha + 2)
            * (3 + alpha)
            * (4 + alpha)
            * (5 + alpha)
            * (6 + alpha)
        ) / 720
        term2 = (
            (alpha + 2)
            * (3 + alpha)
            * (4 + alpha)
            * (5 + alpha)
            * (6 + alpha)
            * (7 + alpha + beta)
            * common_term
        ) / 240
        term3 = (
            (3 + alpha)
            * (4 + alpha)
            * (5 + alpha)
            * (6 + alpha)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * common_term**2
        ) / 192
        term4 = (
            (4 + alpha)
            * (5 + alpha)
            * (6 + alpha)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * common_term**3
        ) / 288
        term5 = (
            (5 + alpha)
            * (6 + alpha)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * (10 + alpha + beta)
            * common_term**4
        ) / 768
        term6 = (
            (6 + alpha)
            * (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * (10 + alpha + beta)
            * (11 + alpha + beta)
            * common_term**5
        ) / 3840
        term7 = (
            (7 + alpha + beta)
            * (8 + alpha + beta)
            * (9 + alpha + beta)
            * (10 + alpha + beta)
            * (11 + alpha + beta)
            * (12 + alpha + beta)
            * common_term**6
        ) / 46080

        return term1 + term2 + term3 + term4 + term5 + term6 + term7
    elif n > 6:
        raise ValueError(
            f"The current implementation supports a maximum degree of 6, but you entered {n}. Higher degrees may lead to numerical instabilities, overfitting, and increased computational complexity. Please consider using a lower degree."
        )
    elif n <= 0:
        raise ValueError(
            "Degrees must be positive. Zero or Negative degrees are not allowed."
        )


def shifted_jacobi_polynomial(x, n, alpha, beta, zeta, a, b, backend):
    if n == 1:
        return (
            alpha - beta + (alpha + beta + 2) * (2 * x**zeta - a - b) / (b - a)
        ) / 2
    elif n == 2:
        return (
            ((alpha + 1) * (alpha + 2)) / 2
            + (
                (alpha + 2)
                * (3 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 2
            + (
                (3 + alpha + beta)
                * (4 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 8
        )
    elif n == 3:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha)) / 6
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 4
            + (
                (3 + alpha)
                * (4 + alpha + beta)
                * (5 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 8
            + (
                (4 + alpha + beta)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 48
        )
    elif n == 4:
        return (
            ((alpha + 1) * (alpha + 2) * (3 + alpha) * (4 + alpha)) / 24
            + (
                (alpha + 2)
                * (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 12
            + (
                (3 + alpha)
                * (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 16
            + (
                (4 + alpha)
                * (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 48
            + (
                (5 + alpha + beta)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 384
        )
    elif n == 5:
        return (
            ((alpha + 1) * (alpha + 2) * (alpha + 3) * (alpha + 4) * (alpha + 5)) / 120
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 48
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 48
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 96
            + (
                (alpha + 5)
                * (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 384
            + (
                (6 + alpha + beta)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 5
            )
            / 3840
        )
    elif n == 6:
        return (
            (
                (alpha + 1)
                * (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
            )
            / 720
            + (
                (alpha + 2)
                * (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1)
            )
            / 240
            + (
                (alpha + 3)
                * (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 2
            )
            / 192
            + (
                (alpha + 4)
                * (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 3
            )
            / 288
            + (
                (alpha + 5)
                * (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 4
            )
            / 768
            + (
                (6 + alpha)
                * (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 5
            )
            / 3840
            + (
                (7 + alpha + beta)
                * (8 + alpha + beta)
                * (9 + alpha + beta)
                * (10 + alpha + beta)
                * (11 + alpha + beta)
                * (12 + alpha + beta)
                * ((2 * x**zeta - a - b) / (b - a) - 1) ** 6
            )
            / 46080
        )
    elif n > 6:
        raise ValueError(
            f"The current implementation supports a maximum degree of 6, but you entered {n}. Higher degrees may lead to numerical instabilities, overfitting, and increased computational complexity. Please consider using a lower degree."
        )
    elif n <= 0:
        raise ValueError(
            "Degrees must be positive. Zero or Negative degrees are not allowed."
        )
