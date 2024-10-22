import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer, BertForSequenceClassification
import transformers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm

from datetime import datetime
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


def get_data_from_dataset(path_to_csv):
    def get_one_level_labels(field):
        new_field_title = field + "_proc"
        df[new_field_title] = df[field].apply(literal_eval)
        mlb = MultiLabelBinarizer()
        sparce_labels = mlb.fit_transform(df[new_field_title])
        counter = np.sum(sparce_labels, axis=0)
        # print(counter.shape)
        print(counter.tolist())
        level_classes = mlb.classes_
        print(f"Unique {field} classes: {len(level_classes)}")

        return level_classes, sparce_labels

    df = pd.read_csv(path_to_csv, sep="\t")
    l1_classes, l1_labels = get_one_level_labels("RGNTI_L1")
    l2_classes, l2_labels = get_one_level_labels("RGNTI_L2")

    return df["data"].to_numpy(), l1_classes, l1_labels, l2_classes, l2_labels

def get_data_from_dataset_common(path_to_train_csv, path_to_test_csv):
    df_train = pd.read_csv(path_to_train_csv, sep="\t")
    df_test = pd.read_csv(path_to_test_csv, sep="\t")

    df_concat = pd.concat([df_train, df_test], ignore_index=True)

    def one_level_encode(df_concat, level_column):
        df_concat[level_column + "_proc"] = df_concat[level_column].apply(literal_eval)
        mlb = MultiLabelBinarizer()
        sparce_labels = mlb.fit_transform(df_concat[level_column + "_proc"])
        print(f"{level_column} classes: {mlb.classes_}")
        return mlb.classes_, sparce_labels
    
    l1_classes, l1_labels = one_level_encode(df_concat, "RGNTI_L1")
    l2_classes, l2_labels = one_level_encode(df_concat, "RGNTI_L2")

    def get_splitted_data_from_level(text, labels, classes, threshold):
        return {"train_text": text[:threshold],
                "train_labels": labels[:threshold],
                "test_text": text[threshold:],
                "test_labels": labels[threshold:],
                "classes": classes}
    
    train_threshold = len(df_train)
    l1_data = get_splitted_data_from_level(df_concat["data"].to_numpy(), l1_labels, l1_classes, train_threshold)
    l2_data = get_splitted_data_from_level(df_concat["data"].to_numpy(), l2_labels, l2_classes, train_threshold)

    class_weights, class_weights_dict = compute_class_weights(df_train)

    return l1_data, l2_data, class_weights_dict


class TextsDataset(Dataset):
    def __init__(self, tokenizer, texts, labels_l1, labels_l2):
        self.texts = texts
        self.labels_l1 = labels_l1
        self.labels_l2 = labels_l2
        self.tokenizer = tokenizer

    def __len__(self):
        return self.labels_l1.shape[0]

    def __getitem__(self, idx):

        inputs = self.tokenizer.encode_plus(
            self.texts[idx],
            None,
            pad_to_max_length=True,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label_l1 = torch.from_numpy(self.labels_l1[idx]).to(dtype=torch.float32)
        label_l2 = torch.from_numpy(self.labels_l2[idx]).to(dtype=torch.float32)
        return {
            'ids': inputs["input_ids"].squeeze(),
            'mask': inputs["attention_mask"].squeeze(),
            'token_type_ids': inputs["token_type_ids"].squeeze(),
        }, label_l1, label_l2


class OneLevelBertClassifier(nn.Module):
    def __init__(self, embedding_model_path, n_classes: int, freeze_bert=False):
        super().__init__()

        self.embedder = BertModel.from_pretrained(embedding_model_path)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, n_classes)

        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for param in self.embedder.parameters():
                param.requires_grad = False

    def set_eval(self):
        self.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False

    def set_train(self):
        self.train()
        if self.freeze_bert:
            for param in self.embedder.parameters():
                param.requires_grad = False
        else:
            for param in self.embedder.parameters():
                param.requires_grad = True

    def forward(self, ids, mask, token_type_ids):
        x = self.embedder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.dropout(x.pooler_output)
        logits = self.classifier(x)

        return logits


class TwoBertClassifier(nn.Module):
    def __init__(self, embedding_model_path, l1_classes: int, l2_classes: int, freeze_bert=False):
        super().__init__()

        self.embedder_l1 = BertModel.from_pretrained(embedding_model_path)
        self.embedder_l2 = BertModel.from_pretrained(embedding_model_path)

        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for param in self.embedder_l1.parameters():
                param.requires_grad = False
            for param in self.embedder_l2.parameters():
                param.requires_grad = False

        self.dropout_l1 = nn.Dropout(0.2)
        self.classifier_l1 = nn.Linear(768, l1_classes)

        self.dropout_l2 = nn.Dropout(0.2)
        self.classifier_l2 = nn.Linear(768 + l1_classes, l2_classes)

    def set_eval(self):
        self.eval()
        for param in self.embedder_l1.parameters():
            param.requires_grad = False

        for param in self.embedder_l2.parameters():
            param.requires_grad = False

    def set_train(self):
        self.train()
        if self.freeze_bert:
            for param in self.embedder_l1.parameters():
                param.requires_grad = False
            for param in self.embedder_l2.parameters():
                param.requires_grad = False
        else:
            for param in self.embedder_l1.parameters():
                param.requires_grad = True
            for param in self.embedder_l2.parameters():
                param.requires_grad = True

    def forward(self, ids, mask, token_type_ids):
        x = self.embedder_l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.dropout_l1(x.pooler_output)
        l1_logits = self.classifier_l1(x)

        x = self.embedder_l2(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.dropout_l2(x.pooler_output)
        x = torch.cat((x, l1_logits), dim=1)
        l2_logits = self.classifier_l2(x)

        return l1_logits, l2_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (1 - pt) ** self.gamma * bce_loss
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        loss = loss.mean()
        return loss


def compute_class_weights(df, level_column='RGNTI_L1'):
    # Flatten the 'keys' column
    df[level_column + "_proc"] = df[level_column].apply(literal_eval)
    all_labels = [label for sublist in df[level_column+"_proc"] for label in sublist]

    # Get unique classes
    unique_classes = np.unique(all_labels)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=all_labels
    )

    # Create a dictionary mapping from class to weight
    class_weights_dict = dict(zip(unique_classes, class_weights))

    print("Class weights:", class_weights_dict)
    return class_weights, class_weights_dict


if __name__ == "__main__":
    epoch_num = 7
    device = "cuda"
    batch_size = 16
    initial_lr = 1e-5
    resume_path = ""

    # Create folders
    experiment_path = Path(f"exps/experiment_{int(time.time())}")
    experiment_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = experiment_path.joinpath("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    log_path = experiment_path.joinpath("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path.absolute().as_posix())

    # Calculate classes
    print("Read labels")
    l1_data, l2_data, class_weights_dict = get_data_from_dataset_common("teach_slice_80_l2_drop_wasted_v2.csv",
                                                                        "test_slice_20_l2_drop_wasted_v2.csv")

    # Instant model, optimizer, etc.
    print("Creating model")
    embedding_model_name = "miemBertProject/miem-scibert-linguistic"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    # model = TwoBertClassifier(embedding_model_name, l1_classes=len(l1_data["classes"]), l2_classes=len(l2_data["classes"]), freeze_bert=False)
    model = OneLevelBertClassifier(embedding_model_name, n_classes=len(l1_data["classes"]), freeze_bert=False)
    if resume_path:
        print(f"Loading model from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    print(model)
    with open(experiment_path.joinpath("model_structure.txt"), "w") as f:
        f.writelines(str(model))
    model.to(device)

    class_weights = list(class_weights_dict.values())
    loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(class_weights, device='cuda', dtype=torch.float))

    # loss_fn = FocalLoss()
    loss_fn.to(device)
    optimizer = transformers.AdamW(model.parameters(), lr=initial_lr, weight_decay = 0.01)

    # Create Datasets and Dataloaders
    training_data = TextsDataset(tokenizer, l1_data["train_text"], l1_data["train_labels"], l2_data["train_labels"])
    test_data = TextsDataset(tokenizer, l1_data["test_text"], l1_data["test_labels"], l2_data["test_labels"])

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total_steps = len(train_dataloader)*epoch_num
    warm_up_end = 2000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=initial_lr,
                                                    total_steps=total_steps,
                                                    pct_start=warm_up_end/total_steps,
                                                    div_factor=1e2)

    min_val_loss = np.inf
    max_val_acc = 0
    max_val_f1 = 0
    print("Start training")
    for epoch in range(epoch_num):
        model.set_train()
        total_loss = 0
        total_l1_loss = 0
        total_l2_loss = 0
        train_bar = tqdm(train_dataloader, ncols=200)

        # Train epoch
        for step, (input, true_l1_labels, true_l2_labels) in enumerate(train_bar, 1):
            true_l1_labels = true_l1_labels.to(device)
            # true_l2_labels = true_l2_labels.to(device)

            ids = input["ids"].to(device)
            mask = input["mask"].to(device)
            token_type_ids = input["token_type_ids"].to(device)

            pred_l1_labels = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids)

            loss = loss_fn(pred_l1_labels, true_l1_labels)
            # loss_l2 = loss_fn(pred_l2_labels, true_l2_labels)
            # loss = loss_l1 + loss_l2

            total_loss += loss.item()
            # total_l1_loss += loss_l1.item()
            # total_l2_loss += loss_l2.item()

            train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch+1, epoch_num, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            train_bar.postfix = f'train_loss={total_loss / step:.4f}'

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)
            optimizer.step()
            scheduler.step()
            if step % 1500 == 0:
                writer.add_scalars('train_loss', {'train_loss': total_loss / step}, step+epoch*len(train_dataloader))
                writer.add_scalars('lr', {'lr': optimizer.param_groups[0]['lr']}, step+epoch*len(train_dataloader))

        # Validation
        model.set_eval()
        total_loss = 0

        l1_total_loss = 0
        l1_total_acc = 0
        l1_total_f1 = 0
        l1_total_f1_macro = 0
        l1_total_f1_weighted = 0

        # l2_total_loss = 0
        # l2_total_acc = 0
        # l2_total_f1 = 0
        # l2_total_f1_macro = 0
        # l2_total_f1_weighted = 0

        validation_bar = tqdm(test_dataloader, ncols=220)
        with torch.no_grad():
            for step, (input, true_l1_labels, true_l2_labels) in enumerate(validation_bar, 1):
                true_l1_labels = true_l1_labels.to(device)
                # true_l2_labels = true_l2_labels.to(device)

                ids = input["ids"].to(device)
                mask = input["mask"].to(device)
                token_type_ids = input["token_type_ids"].to(device)

                pred_l1_labels = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids)

                loss = loss_fn(pred_l1_labels, true_l1_labels)
                # loss_l2 = loss_fn(pred_l2_labels, true_l2_labels)
                # loss = loss_l1 + loss_l2

                total_loss += loss.item()
                # l1_total_loss += loss_l1.item()
                # l2_total_loss += loss_l2.item()

                validation_bar.desc = 'validate[{}/{}][{}]'.format(
                    epoch+1, epoch_num, datetime.now().strftime("%Y-%m-%d-%H:%M"))

                def calculate_level_metrics(pred_labels, true_labels):
                    pred_labels_cpu = torch.sigmoid(pred_labels).detach().cpu().numpy()
                    pred_labels_cpu = np.where(pred_labels_cpu<0.5, 0, 1)

                    true_labels_cpu = true_labels.detach().cpu().numpy()
                    accuracy = accuracy_score(y_true=true_labels_cpu, y_pred=pred_labels_cpu)
                    f1 = f1_score(y_true=true_labels_cpu, y_pred=pred_labels_cpu, average="micro")
                    f1_macro = f1_score(y_true=true_labels_cpu, y_pred=pred_labels_cpu, average="macro")
                    f1_weighted = f1_score(y_true=true_labels_cpu, y_pred=pred_labels_cpu, average="weighted")
                    return accuracy, f1, f1_macro, f1_weighted


                accuracy, f1, f1_macro, f1_weighted = calculate_level_metrics(pred_l1_labels, true_l1_labels)
                l1_total_acc += accuracy
                l1_total_f1 += f1
                l1_total_f1_macro += f1_macro
                l1_total_f1_weighted += f1_weighted

                # accuracy, f1, f1_macro, f1_weighted = calculate_level_metrics(pred_l2_labels, true_l2_labels)
                # l2_total_acc += accuracy
                # l2_total_f1 += f1
                # l2_total_f1_macro += f1_macro
                # l2_total_f1_weighted += f1_weighted

                validation_bar.postfix = (f'valid_loss={total_loss / step:.4f}, '
                                          f'l1_acc={(l1_total_acc / step) * 100:.3f}, '
                                          # f'l2_acc={(l2_total_acc / step) * 100:.3f}, '
                                          f'l1_f1_weighted={(l1_total_f1_weighted / step) * 100:.2f}, '
                                          # f'l2_f1_weighted={(l2_total_f1_weighted / step) * 100:.2f}, '
                                          f'l1_f1_micro={(l1_total_f1 / step) * 100:.2f}, '
                                          f'l1_f1_macro={(l1_total_f1_macro / step) * 100:.2f}')

        epoch_val_loss = total_loss / step
        epoch_val_acc = l1_total_acc / step
        epoch_val_f1 = l1_total_f1 / step
        epoch_val_f1_macro = l1_total_f1_macro / step
        epoch_val_f1_weighted = l1_total_f1_weighted / step
        writer.add_scalars(
            'val_loss', {'val_loss': epoch_val_loss,
                         'acc': epoch_val_acc,
                         'f1': epoch_val_f1,
                         'f1_macro': epoch_val_f1_macro,
                         'f1_weighted': epoch_val_f1_weighted}, epoch)


        # Save checkpoint
        model_dict = model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, checkpoints_path.joinpath(f'model_{str(epoch).zfill(4)}.tar').as_posix())

        if epoch_val_loss < min_val_loss:
            min_val_loss = epoch_val_loss
            print(f'New best vall loss = {min_val_loss}')

        if epoch_val_acc > max_val_acc:
            max_val_acc = epoch_val_acc
            torch.save(state_dict, checkpoints_path.joinpath('best_val_acc_model.tar').as_posix())
            print(f'New best vall acc = {max_val_acc} achived on {epoch} epoch. Model saved.')

        if epoch_val_f1 > max_val_f1:
            max_val_f1 = epoch_val_f1
            print(f'New best vall f1 = {max_val_f1}')

        # scheduler.step()

    print(f"Best acc: {max_val_acc}, best f1: {max_val_f1}, best val loss: {min_val_loss}")