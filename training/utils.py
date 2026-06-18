import torch

def accumulate_gradients(loss, optimizer, model, accumulation_steps, step, total_steps, max_grad_norm, scheduler):
    loss = loss / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()