pub struct Timestamp(u64);

pub trait InTimestampUnits {
    /// milliseconds
    fn ms(self) -> Timestamp;

    /// microseconds
    fn us(self) -> Timestamp;

    /// nanoseconds
    fn ns(self) -> Timestamp;
}

impl InTimestampUnits for f64 {
    fn ms(self) -> Timestamp {
        Timestamp((self * 1_000_000.0) as u64)
    }
    fn us(self) -> Timestamp {
        Timestamp((self * 1_000.0) as u64)
    }
    fn ns(self) -> Timestamp {
        Timestamp(self as u64)
    }
}

impl InTimestampUnits for u64 {
    fn ms(self) -> Timestamp {
        Timestamp(self * 1_000_000u64)
    }
    fn us(self) -> Timestamp {
        Timestamp(self * 1_000u64)
    }
    fn ns(self) -> Timestamp {
        Timestamp(self)
    }
}
