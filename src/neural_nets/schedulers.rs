pub trait Scheduler {}
#[derive(Debug, Clone)]
pub struct ConstantScheduler;

impl Scheduler for ConstantScheduler {}
#[derive(Debug, Clone)]
pub struct ExponentialScheduler;
impl Scheduler for ExponentialScheduler {}
#[derive(Debug, Clone)]
pub struct NoamScheduler {}

impl Scheduler for NoamScheduler {}
#[derive(Debug, Clone)]
pub struct KingScheduler;
impl Scheduler for KingScheduler {}
#[derive(Debug, Clone)]
pub enum Schedulers {
    Constant(ConstantScheduler),
    Exponential(ExponentialScheduler),
    NoamScheduler(NoamScheduler),
    King(KingScheduler),
}

impl Default for Schedulers {
    fn default() -> Self {
        Self::Constant(ConstantScheduler)
    }
}
