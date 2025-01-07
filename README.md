<div align="center">
    <img src="./misc/icon.svg"/>
</div>

## Cryptocurrency Price Forecasting

This project aims to predict cryptocurrency prices, specifically Bitcoin, using machine learning techniques. With the cryptocurrency market being highly volatile, accurate price prediction is challenging but essential for traders and investors to make informed decisions.

The proposed solution uses an XGBoost regression model trained with popular technical indicators such as **Exponential Moving Average (EMA), Moving Average Convergence Divergence (MACD)**, and **Relative Strength Index (RSI)**. The model is trained with historical Bitcoin price data and evaluated for accuracy using metrics such as **MAE (Mean Absolute Error)**, **RMSE (Root Mean Squared Error)**, and **R²**.

## Reference Paper

The work that underpinned the approach used in this project is described in the following paper:

**Cryptocurrency Price Forecasting Using XGBoost**

_Publication: [Arxiv, 2024]_

This paper proposes a cryptocurrency price prediction technique using the **XGBoost** machine learning model in conjunction with popular technical indicators such as the **Exponential Moving Average (EMA)**, the **Moving Average Convergence Divergence (MACD)**, and the **Relative Strength Index (RSI)**. The study is based on historical analysis of cryptocurrency prices, such as Bitcoin, and validates the approach through simulations and performance metrics.

You can access the full article [here](https://arxiv.org/abs/2407.11786).

## Features

-   I. **Data Collection**: Historical Bitcoin data is automatically downloaded from Yahoo Finance. ([BTC-USD](https://finance.yahoo.com/quote/BTC-USD/))

-   II. **Pre-processing**: Technical indicators are calculated.

-   III. **Model Training**: The [XGBoost](https://xgboost.readthedocs.io/en/stable/) model is trained to predict Bitcoin closing prices.

-   IV. **Evaluation**: The model is evaluated using **MAE**, **RMSE** and **R²** metrics.

-   V. **Hyperparameter Optimization**: Performs hyperparameter search using **Grid Search** techniques.

## Contribution

Contributions are welcome! If you have suggestions for improvements or want to add new features, feel free to open a pull request or issue.

## License

-   Abyss: [Click here to read license](./LICENSE)
