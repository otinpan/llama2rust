use std::time::Instant;

pub struct Timer{
    label: &'static str,
    start: Instant,
}

impl Timer{
    pub fn start(label: &'static str) -> Self{
        Self{
            label,
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u128{
        self.start.elapsed().as_millis()
    }

    pub fn finish(&self){
        crate::log_info!("timer: {} took {} ms",self.label,self.elapsed_ms());
    }
}
