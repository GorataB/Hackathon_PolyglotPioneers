# Feature Engineering Report

## Project Context

This project forecasts Botswana monthly food price inflation for January 2024 to December 2024. The modelling dataset combines food price indicators with shipping, fuel, and monetary policy variables that may explain movements in domestic food inflation.

The cleaned modelling file is:

- `Data_Clean/prediction_data.csv`

The final cleaned dataset contains 276 monthly observations from `2001-01` to `2023-12` with no missing values in the modelling columns.

## Raw Data Sources Used

| Dataset | File | Frequency | Coverage in raw file | Purpose |
|---|---|---:|---|---|
| Baltic Dry Index | `Data_Raw/01_baltic_dry_index_daily.csv` | Daily | 2000-01-04 to 2023-12-22 | Proxy for global dry bulk shipping costs |
| Brent crude oil price | `Data_Raw/02_brent_crude_monthly.csv` | Monthly | 2000-01-15 to 2023-12-15 | Proxy for global fuel and transport cost pressure |
| Botswana policy rate | `Data_Raw/03_botswana_policy_rate.csv` | Monthly | 2000-01-01 to 2023-12-01 | Domestic monetary policy and demand-side pressure |
| FAO Botswana consumer prices | `Data_Raw/04_fao_botswana_prices.csv` | Monthly | 2000-01-01 to 2023-12-01 | Target and domestic price index variables |
| Human Capital Project data | `Data_Raw/05_human_capital_project.csv` | Monthly in file | 2000-01-01 to 2023-12-01 | Used for the HCP linkage analysis, not the main forecasting table yet |

## Target Variable

The primary forecast target is:

| Variable | Meaning | Justification |
|---|---|---|
| `FAO_23014` | Food price inflation | This is the direct outcome required by the forecasting task. It measures food inflation and is therefore the dependent variable for the model. |

Rows where `FAO_23014` was missing were removed because the model cannot be trained without the target value. After this filtering step, the final modelling period starts in `2001-01` and ends in `2023-12`.

## Engineered and Selected Variables

### FAO Price Variables

The FAO Botswana price data originally appears in long format with columns such as `Date`, `Item Code`, `Item`, and `Value`. It was transformed into wide format so that each indicator becomes its own modelling column.

| Final variable | Source indicator | Meaning | Justification |
|---|---|---|---|
| `FAO_23012` | Consumer Prices, General Indices (2015 = 100) | General consumer price level | Controls for broad inflation pressure in the economy. Food inflation may move with general inflation because of shared cost drivers and macroeconomic shocks. |
| `FAO_23013` | Consumer Prices, Food Indices (2015 = 100) | Food consumer price index | Captures the underlying food price level. This is closely related to the food inflation target and provides useful trend information. |
| `FAO_23014` | Food price inflation | Monthly food inflation target | Target variable for forecasting. |

Transformation:

- Converted `Date` to monthly `year_month`.
- Pivoted the long FAO data by `Item Code`.
- Renamed indicators to `FAO_23012`, `FAO_23013`, and `FAO_23014`.
- Dropped rows where `FAO_23014` was missing.

### Baltic Dry Index Features

The Baltic Dry Index is a daily global shipping-cost index for dry bulk goods such as grains, minerals, and raw materials. Because food imports and food supply chains can be affected by international shipping costs, BDI was included as an external cost-pressure variable.

The BDI raw file contains:

- `Date`
- `BDI_Close`
- `BDI_High`
- `BDI_Low`

Since the forecasting target is monthly, the daily BDI data was aggregated to monthly level.

| Final variable | Engineering method | Meaning | Justification |
|---|---|---|---|
| `BDI_std` | Monthly standard deviation of daily `BDI_Close` | Monthly volatility in shipping costs | Volatility in shipping costs can signal supply-chain uncertainty and cost shocks that may later influence food prices. |
| `first` | First daily `BDI_Close` in each month | Opening BDI value for the month | Used to calculate the monthly return. |
| `last` | Last daily `BDI_Close` in each month | Closing BDI value for the month | Used to calculate the monthly return. |
| `monthly_return_bdi` | `(last - first) / first` | Monthly percentage change in BDI | Captures direction and size of shipping-cost movement during the month. Rising shipping costs can feed into imported food and input prices. |

Daily to monthly aggregation choices:

- Monthly volatility was calculated using the standard deviation of daily close values within each month.
- Monthly return was calculated using the first and last observed close values in each month.
- The month key was represented as `year_month` in `YYYY-MM` format to align with other monthly datasets.

Reason for using close values:

- `BDI_Close` represents the final daily index value and is a stable basis for monthly volatility and return calculations.
- High and low values were inspected but not included in the final cleaned table. They can be added later as monthly range features if model validation shows benefit.

### Brent Crude Oil Feature

| Final variable | Source | Meaning | Justification |
|---|---|---|---|
| `Brent_USD_per_barrel` | Brent crude monthly data | Global oil price in USD per barrel | Oil prices affect transport, production, and import costs. Food prices may respond directly through transport costs and indirectly through input costs such as fertilizer and logistics. |

Transformation:

- Converted `Date` to `year_month`.
- Kept the monthly Brent price as a numeric explanatory variable.
- Merged on `year_month`.

### Botswana Policy Rate Feature

| Final variable | Source | Meaning | Justification |
|---|---|---|---|
| `policy_rate` | Botswana policy rate data | Domestic policy interest rate | Captures monetary policy conditions. Higher policy rates may reduce demand-side inflation pressure, while rate changes may also respond to existing inflation trends. |

Transformation:

- Converted `Date` to `year_month`.
- Kept the monthly policy rate as a numeric explanatory variable.
- Merged on `year_month`.

## Merge Strategy

All datasets were aligned using a common monthly key:

- `year_month`
- Format: `YYYY-MM`

The merge process was:

1. Convert all date columns to datetime.
2. Create a monthly key using the year and month.
3. Transform FAO data from long format to wide format.
4. Aggregate daily BDI data to monthly features.
5. Select monthly Brent and policy rate variables.
6. Merge all datasets on `year_month`.
7. Sort chronologically.
8. Drop rows where the target variable `FAO_23014` is missing.

The modelling table after cleaning contains:

| Column | Role |
|---|---|
| `year_month` | Time index |
| `FAO_23012` | General CPI explanatory variable |
| `FAO_23013` | Food CPI explanatory variable |
| `FAO_23014` | Forecast target |
| `BDI_std` | Shipping-cost volatility feature |
| `first` | Monthly first BDI close |
| `last` | Monthly last BDI close |
| `monthly_return_bdi` | Monthly BDI return |
| `Brent_USD_per_barrel` | Global oil price feature |
| `policy_rate` | Botswana monetary policy feature |

## Missing Data Handling

The final cleaned modelling dataset has no missing values in the selected columns.

The main missing-data decision was to drop rows without `FAO_23014`, because the model requires observed food inflation values during training and validation. This reduced the usable modelling period to `2001-01` through `2023-12`.

## Lag Structures

At the current stage, the cleaned dataset contains contemporaneous monthly variables. For time-series forecasting, lag features should be added before final model training to avoid relying only on same-month information.

Recommended lag features:

| Variable | Suggested lags | Reason |
|---|---:|---|
| `FAO_23014` | 1, 3, 6, 12 months | Captures persistence, short-term momentum, seasonality, and annual inflation patterns. |
| `FAO_23013` | 1, 3, 6, 12 months | Food CPI level may lead or explain food inflation changes. |
| `FAO_23012` | 1, 3, 6, 12 months | General inflation pressure may influence future food inflation. |
| `BDI_std` | 1, 3, 6 months | Shipping-cost volatility may affect local food prices with delay. |
| `monthly_return_bdi` | 1, 3, 6 months | Shipping-cost increases may pass through to food prices after trade and distribution delays. |
| `Brent_USD_per_barrel` | 1, 3, 6 months | Oil-price shocks may affect food prices through delayed transport and production costs. |
| `policy_rate` | 1, 3, 6, 12 months | Monetary policy effects usually operate with delay. |

Important modelling note:

- Lag features should be created using only past information relative to the forecast month.
- Rolling statistics should also be shifted so that the model does not accidentally use future data.

## Additional Transformations Recommended

The following transformations should be considered during model development:

| Transformation | Variables | Reason |
|---|---|---|
| Month-of-year indicators | `year_month` | Captures seasonality in food prices. |
| Rolling means | `FAO_23014`, `FAO_23013`, `Brent_USD_per_barrel`, `BDI_std` | Smooths noisy monthly movements and captures recent trend. |
| Rolling standard deviation | `FAO_23014`, `BDI_std`, `Brent_USD_per_barrel` | Captures volatility and instability. |
| Percentage change | `FAO_23012`, `FAO_23013`, `Brent_USD_per_barrel` | Converts levels into growth rates. |
| Scaling/standardisation | Model input features | Needed for neural network models and useful for models sensitive to variable scale. |

## Cross-Dataset Integration Rationale

The final feature set integrates domestic price data, global shipping data, global fuel prices, and domestic monetary policy.

This integration is important because food inflation is affected by multiple channels:

- Domestic inflation conditions through general CPI and food CPI.
- Imported cost pressure through Brent crude and the Baltic Dry Index.
- Supply-chain stress through BDI volatility and monthly return.
- Policy and macroeconomic conditions through the Botswana policy rate.

Using multiple datasets should improve the model's ability to capture realistic drivers of food inflation instead of relying only on historical food inflation.

## Limitations and Next Improvements

Current limitations:

- The cleaned dataset currently includes only contemporaneous features.
- External human capital indicators are not yet integrated into the forecasting table.
- BDI high and low values were not included as separate monthly range features.
- The final lag structure still needs to be selected and validated.

Recommended next improvements:

1. Add lag features for target and explanatory variables.
2. Add month-of-year seasonality features.
3. Add rolling mean and rolling volatility features.
4. Test whether BDI monthly range features improve validation RMSE.
5. Keep a feature-selection log showing which features were kept or removed and why.

## Conclusion

The feature engineering process created a monthly modelling table that combines food inflation, consumer price indices, shipping-cost volatility, shipping-cost returns, oil prices, and policy rates. The most important engineering step was converting the daily Baltic Dry Index into monthly features that can be merged with the monthly food inflation target.

The current dataset is suitable as a baseline modelling table. The next step is to add lagged and rolling features, train the classical and deep learning models, and document which feature set produces the strongest validation performance.
