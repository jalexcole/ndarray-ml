use std::fmt::Display;

pub mod kernels;

struct PQNode {}

struct PriorityQueue {}

struct BallTreeNode;

struct BallTree;

struct DiscreteSampler;

trait KernelBase: Display {}

struct LinearKernel;

struct PolyKernel;

struct RBFKernel;

struct KernelInitializer;


pub enum Either<L, R> {
    Left(L),
    Right(R),
} 