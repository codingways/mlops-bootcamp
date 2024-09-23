CREATE TABLE IF NOT EXISTS raw_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE,
    truck_id INTEGER,
    total_weight FLOAT,
    UNIQUE (truck_id, date)  -- Agregar restricción de clave única
);

CREATE INDEX idx_raw_data_truck_id ON raw_data(truck_id);
CREATE INDEX idx_raw_data_date ON raw_data(date);
