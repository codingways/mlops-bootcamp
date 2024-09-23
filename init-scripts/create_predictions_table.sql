CREATE TABLE IF NOT EXISTS predictions (
    date DATE,
    truck_id INTEGER,
    predicted_weight FLOAT,
    lower_bound FLOAT,
    upper_bound FLOAT
);
