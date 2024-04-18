import wandb
import transformer, bpe_tokenizer, train, loss, optimizer, utils
import numpy as np
import torch


def get_data(path, batch_size, context_length, device, num_batches=None):
    corpus = np.memmap(path, dtype=np.uint16, mode="r")
    if num_batches is None:
        while True:
            data_loader = utils.dataloader(corpus, batch_size, context_length, device)
            yield data_loader
    else:
        for _ in range(num_batches):
            data_loader = utils.dataloader(corpus, batch_size, context_length, device)
            yield data_loader

def main():
    train_path = "/data/TinyStoriesV2-GPT4-train.bin"
    valid_path = "/data/TinyStoriesV2-GPT4-valid.bin"
    """config_dict = {
            "vocab_size" : 10000, 
            "context_length" : 256,
            "d_model" : 512,
            "d_ff" : 2048,
            "num_layers" : 4, 
            "num_heads" : 16,
            "step_count" : 20000,
            "learning_rate" : 1e-3,
            "attn_pdrop" : 0.1,
            "attn_pdrop" : 0.1,
            "residual_pdrop" : 0.1,
            "batch_size" : 64
                }"""
    
    #sweep_id = wandb.sweep(sweep_config, project='small_model', entity='jjian', function="train")
    wandb.init(project="lr_sweep_tinystories")
    #wandb.config.update(config_dict)
    
    device = torch.device("cuda:0")
    print(device)

    config = wandb.config
    model = transformer.Transformer(
                    config.vocab_size, config.context_length, config.d_model, 
                    config.num_layers, config.num_heads, config.d_ff, 
                    attn_pdrop=config.attn_pdrop, residual_pdrop=config.residual_pdrop)
    model = model.to(device)
    step_count = 327680000 // (config.context_length * config.batch_size)
    #model = torch.compile(model)
    optim = optimizer.AdamW(model.parameters(), lr=config.learning_rate)
    train_loader = get_data(train_path, config.batch_size, config.context_length, device)
    valid_loader = get_data(valid_path, config.batch_size, config.context_length, device)
    criterion = loss.CrossEntropyLoss()
    final_loss = train.train(model, train_loader, valid_loader, optim, criterion, step_count, device, checkpoints="/home/c-jjian/assignments/spring2024-assignment1-basics/results/train_exp_tiny_sweep/")
    return final_loss

if __name__ == "__main__":
    # Start the sweep
    sweep_config = {"method" : "grid", 
                    "parameters" : {
                        "vocab_size" : {"value" : 10000}, 
                        "context_length" : {"value" : 256},
                        "d_model" : {"value" : 512},
                        "d_ff" : {"value" : 2048},
                        "num_layers" : {"value" : 4}, 
                        "num_heads" : {"value" : 16},
                        #"step_count" : {"value" : 20000},
                        "learning_rate" : {"value" : 1e-3},
                        "attn_pdrop" : {"value" : 0.1},
                        "residual_pdrop" : {"value" : 0.1},
                        "batch_size" : {"values" : [128, 256, 512, 1024]}
                        }
                    }
    sweep_id = wandb.sweep(sweep_config, project='batch_sweep_tinystories', entity='jjian')

    wandb.agent(sweep_id, function=main, count=10)
