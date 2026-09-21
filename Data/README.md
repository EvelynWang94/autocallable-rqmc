# Market data

Raw Bloomberg workbooks are not distributed. The CPU example and regression
tests use synthetic inputs and run without them.

To rerun the original market-calibrated notebooks, place your authorised copy
at `Data/Bloomberg_Real_Autocallable_Market_Data_HSBC.xlsx`. The frozen SHA-256 is:

```
CD84790399F9CDA5C5B0ED3E95D973216314CA54DD4BDBA6D9D22C399DBD49E6
```

The notebook hash checks deliberately reject other workbooks. Do not bypass
them and then label a different calibration as a reproduction of the report.
Files placed in this directory are ignored by Git. Contract definitions are
included in `config/`; derived experimental evidence is included in `results/`.
