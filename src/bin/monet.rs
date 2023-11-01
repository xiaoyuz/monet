use anyhow::Result;

use clap::Parser;
use monet::{trans::TransformComponent, Args};

fn run(args: Args) -> Result<()> {
    let component = TransformComponent::new(args)?;
    component.run()
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
