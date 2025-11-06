claude-flow swarm "Perform machine learning engineering on data/rossmann-store-sales/train.csv to predict Sales using MLE-STAR methodology..." --claude

# claude-flow automation mle-star --dataset data/rossmann-store-sales/train.csv --target Sales --claude

# claude-flow automation mle-star \
#   --dataset data/rossmann-store-sales/train.csv \
#   --target Sales \
#   --output models/ \
#   --name "sales-prediction" \
#   --search-iterations 5 \
#   --refinement-iterations 8 \
#   --max-agents 8 \
#   --claude