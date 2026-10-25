# Data

## Included

`processed/` contains the final 500 m modelling dataset (`london_pm25_final.parquet` / `.csv`, 6,670 cells) and the grid (`london_500m_grid.gpkg`), so the modelling scripts run without the raw files.

## Raw data (not included)

Download from the London Datastore into `data/raw/` with these file names:

| File | Source |
|---|---|
| `LAEI2022-Concentrations-Data-CSV.zip` | https://data.london.gov.uk/dataset/london-atmospheric-emissions-inventory-laei-2022-2lg5g |
| `LAEI2022-Traffic-Flows-and-Speeds-Major-and-Minor-Roads-GIS.zip` | same LAEI 2022 page |
| `Green_Cover_2024.zip` | https://data.london.gov.uk/dataset/green-cover-2024-e56n0 |
| `London_Boroughs.gpkg` | https://data.london.gov.uk/dataset/london-boroughs-e55pw |

The raw files total over 600 MB, so they are not stored on GitHub. The map scripts also need `London_Boroughs.gpkg`.

Data © Greater London Authority, made available under the Open Government Licence v3.0 (check each dataset page).
