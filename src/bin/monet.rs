use std::sync::Arc;

use anyhow::Result;

use clap::Parser;
use monet::{trans::TransformComponent, Args};
use tokio::{join, spawn, sync::Semaphore};

async fn run(args: Args) -> Result<()> {
    let component = TransformComponent::new(&args)?;
    let component = Arc::new(component);

    let semaphore = Arc::new(Semaphore::new(args.num_threads));

    let mut tasks = vec![];
    for idx in 0..args.num_samples {
        let permit = semaphore.clone().acquire_owned().await?;
        let c = component.clone();
        let handle = spawn(async move {
            println!("sample {} start.", idx + 1);
            let res = c.run(idx).await;
            drop(permit);
            res
        });
        tasks.push(handle);
    }

    join!(async {
        for handle in tasks {
            let _ = handle.await;
        }
    });
    println!("All samples finished.");
    Ok(())
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let args = Args::parse();
    run(args).await
}
