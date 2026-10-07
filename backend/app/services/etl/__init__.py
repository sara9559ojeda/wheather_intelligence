"""ETL: carga de datos crudos (data/raw) a PostgreSQL.

  - ``location_seed``    : crea/actualiza la fila de la ubicación en ``locations``
  - ``historical_loader``: valida y carga el parquet histórico en ``weather_observations``
  - ``load_historical``  : CLI  (python -m backend.app.services.etl.load_historical)
"""
