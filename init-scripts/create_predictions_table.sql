CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    date DATE,
    truck_id INTEGER,
    predicted_weight FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT,
    UNIQUE (truck_id, date)  -- Agregar restricción de clave única
);

CREATE INDEX idx_predictions_truck_id ON predictions(truck_id);
CREATE INDEX idx_predictions_date ON predictions(date);
