"""Adquisición de datos meteorológicos (fase 2 — Data Understanding).

Módulos:
  - ``open_meteo_archive``  : histórico horario ERA5 vía Open-Meteo (dataset principal)
  - ``meteostat_station``   : observaciones reales de estación (contraste)
  - ``metadata``            : escritura de los *sidecars* ``*.meta.json``
  - ``download_historical`` : CLI que produce data/raw/open_meteo_<slug>_hourly.parquet
  - ``download_meteostat``  : CLI que produce data/raw/meteostat_<station>_hourly.parquet
"""
