import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from scipy.optimize import linear_sum_assignment
import wandb

from .build import MODELS
from utils.logger import *
from extensions.chamfer_distance import ChamferDistance
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.transformer import Group, ZGroup, PatchEmbedding, PositionEmbeddingCoordsSine, GPTExtractor, \
    GPTGenerator, MAEExtractor, MAEGenerator


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config):
        super(MaskTransformer, self).__init__()

        self.embed_dim = config.embed_dim
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.with_color = config.with_color
        self.input_channel = 6 if self.with_color else 3
        self.img_queries = config.img_queries
        self.text_queries = config.text_queries
        self.global_query_num = self.img_queries + self.text_queries
        self.mask_type = config.mask_type
        self.mask_ratio = config.mask_ratio
        self.stop_grad = config.stop_grad

        self.embed = PatchEmbedding(embed_dim=self.embed_dim, input_channel=self.input_channel,
                                    large=config.large_embedding)

        print_log(f'[ReCon] divide point cloud into G{config.num_group} x S{config.group_size} points ...',
                  logger='ReCon')

        if self.mask_type == 'causal':
            self.group_divider = ZGroup(num_group=config.num_group, group_size=config.group_size)
            self.encoder = GPTExtractor(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                depth=config.depth,
                group_size=config.group_size,
                drop_path_rate=config.drop_path_rate,
                stop_grad=self.stop_grad,
                pretrained_model_name=config.pretrained_model_name,
            )
            self.decoder = GPTGenerator(
                embed_dim=config.embed_dim,
                depth=config.decoder_depth,
                drop_path_rate=config.drop_path_rate,
                num_heads=config.num_heads,
                group_size=config.group_size,
                input_channel=self.input_channel
            )
            self.pos_embed = PositionEmbeddingCoordsSine(3, self.embed_dim, 1.0)

        else:
            self.group_divider = Group(num_group=config.num_group, group_size=config.group_size)
            self.encoder = MAEExtractor(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                depth=config.depth,
                group_size=config.group_size,
                drop_path_rate=config.drop_path_rate,
                stop_grad=self.stop_grad,
                pretrained_model_name=config.pretrained_model_name,
            )
            self.decoder = MAEGenerator(
                embed_dim=config.embed_dim,
                depth=config.decoder_depth,
                drop_path_rate=config.drop_path_rate,
                num_heads=config.num_heads,
                group_size=config.group_size,
                input_channel=self.input_channel
            )
            self.pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.embed_dim)
            )
            self.decoder_pos_embed = nn.Sequential(
                nn.Linear(3, 128),
                nn.GELU(),
                nn.Linear(128, self.embed_dim)
            )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.global_query = nn.Parameter(torch.zeros(1, self.global_query_num, self.embed_dim))
        self.apply(self._init_weights)

        # do not perform additional mask on the first (self.keep_attend) tokens
        self.keep_attend = 10
        self.num_group = config.num_group
        self.num_mask = int((self.num_group - self.keep_attend) * self.mask_ratio)

        if config.pretrained_model_name == "":
            print_log(f'[ReCon] No pretrained model is loaded.', logger='ReCon')
        elif config.pretrained_model_name in timm.list_models(pretrained=True):
            self.encoder.blocks.load_pretrained_timm_weights()
            print_log(f'[ReCon] Timm pretrained model {config.pretrained_model_name} is successful loaded.',
                      logger='ReCon')
        else:
            print_log(f'[ReCon] Pretrained model {config.pretrained_model_name} is not found in Timm.', logger='ReCon')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _mask_center_rand(self, center):
        """
            center : B G 3
            --------------
            mask : B G (bool)
        """
        B, G, _ = center.shape
        num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return num_mask, overall_mask.to(center.device)

    def inference(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            group_input_tokens = self.embed(neighborhood)  # B G C
            batch_size, seq_len, C = group_input_tokens.size()

            global_query = self.global_query.expand(batch_size, -1, -1)
            pos = self.pos_embed(center.to(group_input_tokens.dtype))

            mask = torch.full(
                (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
            ).to(torch.bool)
            if self.mask_type == 'causal':
                mask = torch.triu(mask, diagonal=1)
            else:
                mask = None

            local_features, global_features = self.encoder(
                group_input_tokens, pos, mask, global_query)

        return pos, local_features, global_features

    def forward_mae(self, pts):
        neighborhood, center = self.group_divider(pts)
        num_mask, mask = self._mask_center_rand(center)
        group_input_tokens = self.embed(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()
        global_query = self.global_query.expand(batch_size, -1, -1)

        pos = self.pos_embed(center.reshape(batch_size, -1, 3))
        decoder_pos = self.decoder_pos_embed(center.reshape(batch_size, -1, 3))
        x_vis, global_features = self.encoder(
            group_input_tokens, pos, mask, global_query)
        generated_points = self.decoder(
            x_vis, decoder_pos, mask)

        gt_points = neighborhood[mask].reshape(batch_size * num_mask, self.group_size, self.input_channel)

        return generated_points, gt_points, global_features

    def forward_gpt(self, pts):
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.embed(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()

        global_query = self.global_query.expand(batch_size, -1, -1)
        pos_absolute = self.pos_embed(center).to(group_input_tokens.dtype)

        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / (relative_norm + 1e-5)
        position = torch.cat([center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position).to(group_input_tokens.dtype)

        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)

        with torch.no_grad():
            attn_mask = torch.triu(attn_mask, diagonal=1)

            # column wise
            overall_mask = np.hstack([
                np.zeros(self.num_group - self.keep_attend - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(overall_mask)
            overall_mask = np.hstack([
                np.zeros(self.keep_attend),
                overall_mask,
            ])
            overall_mask = torch.from_numpy(overall_mask).to(torch.bool).to(group_input_tokens.device)
            eye_mask = torch.eye(self.num_group, device=group_input_tokens.device, dtype=torch.bool)
            attn_mask = attn_mask | overall_mask.unsqueeze(0) & ~eye_mask

        local_features, global_features = self.encoder(
            group_input_tokens, pos_absolute, attn_mask, global_query)
        generated_points = self.decoder(
            local_features, pos_relative, attn_mask)

        gt_points = neighborhood.reshape(batch_size * self.num_group, self.group_size, self.input_channel)

        return generated_points, gt_points, global_features

    def forward(self, pts):
        if self.mask_type == 'causal':
            generated_points, gt_points, global_query = self.forward_gpt(pts)
        else:
            generated_points, gt_points, global_query = self.forward_mae(pts)

        return generated_points, gt_points, global_query


@MODELS.register_module()
class ReCon2(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[ReCon V2]', logger='ReCon V2')
        self.config = config
        self.embed_dim = config.embed_dim
        self.with_color = config.with_color
        self.img_queries = config.img_queries
        self.text_queries = config.text_queries
        self.global_query_num = self.img_queries + self.text_queries
        self.input_channel = 6 if self.with_color else 3
        self.contrast_type = config.contrast_type

        self.model = MaskTransformer(config)
        self.cd_loss = ChamferDistance()
        self.l1_loss = torch.nn.SmoothL1Loss()

        self.img_proj = nn.Linear(self.embed_dim, 1280)
        self.img_proj.apply(self._init_weights)
        self.text_proj = nn.Linear(self.embed_dim, 1280)
        self.text_proj.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def info_nce_loss(self, feat1, feat2, logit_scale=1, mask=None):
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
        all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
        logits = logit_scale * all_feat1 @ all_feat2.T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def distillation_loss(self, token, feature):
        B = token.shape[0]
        loss = 0.0
        for i in range(B):
            pred = token[i]
            feat = feature[i][torch.any(feature[i] != 0, dim=1)]
            feat = F.normalize(feat, dim=-1)
            similarity_matrix = torch.mm(pred, feat.T).cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            loss = loss + self.l1_loss(pred[row_ind], feat[col_ind])

        return loss * 5

    def contrast_loss(self, token, feature):
        if self.contrast_type == 'simclr':
            return self.info_nce_loss(token, feature, logit_scale=self.logit_scale, mask=self.mask)
        elif self.contrast_type == 'byol':
            return self.distillation_loss(token, feature)
        else:
            raise ValueError("Unknown contrast type")

    def inference(self, pts):

        _, encoded_features, global_token = self.model.inference(pts)

        img_token = global_token[:, :self.img_queries]
        img_token = self.img_proj(img_token)
        img_token = F.normalize(img_token, dim=-1)

        text_token = global_token[:, self.img_queries:]
        text_token = self.text_proj(text_token)
        text_token = F.normalize(text_token, dim=-1)

        return encoded_features, global_token, img_token, text_token

    def forward_features(self, pts):

        generated_points, gt_points, global_token = self.model(pts)

        img_token = global_token[:, :self.img_queries]
        img_token = self.img_proj(img_token)
        img_token = F.normalize(img_token, dim=-1)

        text_token = global_token[:, self.img_queries:]
        text_token = self.text_proj(text_token)
        text_token = F.normalize(text_token, dim=-1)

        return img_token, text_token, gt_points, generated_points

    def forward_reconstruct(self, pts):

        _, _, gt_points, generated_points = self.forward_features(pts)

        generated_xyz = generated_points[:, :, :3]
        gt_xyz = gt_points[:, :, :3]
        dist1, dist2, idx = self.cd_loss(generated_xyz, gt_xyz)
        if self.with_color:
            generated_color = generated_points[:, :, 3:]
            gt_color = gt_points[:, :, 3:]
            color_l1_loss = self.l1_loss(generated_color,
                                         torch.gather(gt_color, 1, idx.unsqueeze(-1).expand(-1, -1, 3).long()))
        else:
            color_l1_loss = 0
        cd_l2_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        cd_l1_loss = (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2

        loss = cd_l1_loss + cd_l2_loss + color_l1_loss

        return loss

    def forward_contrast(self, pts, img, text):

        img_token, text_token, _, _ = self.forward_features(pts)
        img_loss = self.contrast_loss(img_token, img)
        text_loss = self.contrast_loss(text_token, text)
        loss = img_loss + text_loss

        return loss

    def forward_all(self, pts, img, text):

        img_token, text_token, gt_points, generated_points = self.forward_features(pts)

        losses = {'mdm': 0, 'csc_img': 0, 'csc_text': 0}

        generated_xyz = generated_points[:, :, :3]
        gt_xyz = gt_points[:, :, :3]
        dist1, dist2, idx = self.cd_loss(generated_xyz, gt_xyz)
        if self.with_color:
            generated_color = generated_points[:, :, 3:]
            gt_color = gt_points[:, :, 3:]
            color_l1_loss = self.l1_loss(generated_color,
                                         torch.gather(gt_color, 1, idx.unsqueeze(-1).expand(-1, -1, 3).long()))
        else:
            color_l1_loss = 0
        cd_l2_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        cd_l1_loss = (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2

        losses['mdm'] = cd_l1_loss + cd_l2_loss + color_l1_loss
        losses['csc_img'] = self.contrast_loss(img_token, img)
        losses['csc_text'] = self.contrast_loss(text_token, text)

        print(losses)
        loss = sum(losses.values())
        return loss

    def forward(self, pts, img, text, type="all"):
        if type == "all":
            return self.forward_all(pts, img, text)
        elif type == "reconstruct":
            return self.forward_reconstruct(pts)
        elif type == "contrast":
            return self.forward_contrast(pts, img, text)
        else:
            raise ValueError("Unknown type")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim
        self.with_color = config.with_color
        self.input_channel =  3
        self.num_group = config.num_group
        self.group_size = config.group_size
        self.img_queries = config.img_queries
        self.text_queries = config.text_queries
        self.global_query_num = self.img_queries + self.text_queries
        self.large_embedding = config.large_embedding
        self.pos_threshold = config.pos_threshold
        self.neg_threshold = config.neg_threshold
        self.num_features = config.num_features

        self.embed = PatchEmbedding(embed_dim=self.embed_dim, input_channel=self.input_channel, large=self.large_embedding)
        self.pos_embed = PositionEmbeddingCoordsSine(3, self.embed_dim, 1.0)

        self.group_divider = ZGroup(num_group=config.num_group, group_size=config.group_size)
        print_log(f'[PointTransformer] divide point cloud into G{config.num_group} x S{config.group_size} points ...',
                  logger='PointTransformer')

        self.encoder = GPTExtractor(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.depth,
            group_size=config.group_size,
            drop_path_rate=config.drop_path_rate,
            stop_grad=False,
        )

        self.decoder = GPTGenerator(
            embed_dim=config.embed_dim,
            depth=config.decoder_depth,
            drop_path_rate=config.drop_path_rate,
            num_heads=config.num_heads,
            group_size=config.group_size,
            input_channel=self.input_channel
        )
        self.global_query = nn.Parameter(torch.zeros(1, self.global_query_num, self.embed_dim))

        feature_dim = 768
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 256),     # First layer to reduce dimensionality
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),     # First layer to reduce dimensionality
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),              # Final embedding layer output size
            nn.BatchNorm1d(64)
        )
        #chamfer distance loss
        self.cd_loss = ChamferDistance()
        self.apply(self._init_weights)

    # def get_loss_acc(self, embeddings, labels, temperature=0.1):
    #     # Normalize embeddings
    #     embeddings = F.normalize(embeddings, dim=-1)

    #     # Get the number of embeddings
    #     batch_size = embeddings.size(0)

    #     # Create a mask for similar pairs
    #     # labels = labels.contiguous().view(-1, 1)
    #     # mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    #     # logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(embeddings.device)
    #     # mask = mask * logits_mask  # Remove self-similarities

    #     labels = labels.contiguous().view(-1, 1)
    #     mask = torch.eq(labels, labels.T).float()
    #     mask.fill_diagonal_(0)

    #     # Compute similarity matrix
    #     sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    #     exp_sim = torch.exp(sim_matrix)
    #     pos_term = (mask * sim_matrix).sum(1)  # Numerator
    #     neg_term = torch.log(exp_sim.sum(1) - exp_sim.diag())

    #     # # Extract positive and negative pairs
    #     # positive_mask = mask.bool()
    #     # negative_mask = (~mask.bool()) * logits_mask.bool()

    #     # positive_pairs = similarity_matrix * positive_mask  # Shape: (batch_size, batch_size)
    #     # negative_pairs = similarity_matrix * negative_mask  # Shape: (batch_size, batch_size)

    #     # # Compute denominators for normalization
    #     # pos_denominator = positive_pairs.sum(dim=1, keepdim=True) + 1e-8
    #     # neg_denominator = negative_pairs.sum(dim=1, keepdim=True) + 1e-8

    #     # # Compute positive loss
    #     # pos_fraction = (positive_pairs + 1e-8) / pos_denominator
    #     # pos_loss = -torch.log(pos_fraction + 1e-8)
    #     # pos_loss = pos_loss.sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)

    #     # # Compute negative loss
    #     # neg_fraction = 1 - (negative_pairs + 1e-8) / neg_denominator
    #     # neg_loss = -torch.log(neg_fraction + 1e-8)
    #     # neg_loss = neg_loss.sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)

    #     # Combine the losses
    #     # loss = pos_loss.mean() + neg_loss.mean()

    #     # # Calculate accuracy
    #     # correct_pos = (positive_pairs > self.pos_threshold).sum().float()
    #     # correct_neg = (negative_pairs < self.neg_threshold).sum().float()

    #     # total_pos = positive_mask.sum()
    #     # total_neg = negative_mask.sum()

    #     # acc = (correct_pos + correct_neg) / (total_pos + total_neg + 1e-8) * 100
    #     loss = -(pos_term - neg_term).mean()
    
    #     # ===== 4. Meaningful Accuracy =====
    #     pos_scores = sim_matrix[mask.bool()].mean()
    #     neg_scores = sim_matrix[~mask.bool()].mean()
    #     acc = (pos_scores > neg_scores).float().item() * 100  # Separation check

    #     return loss, acc

    def get_loss_acc(self, embeddings, labels, temperature=0.1):
        # Normalize embeddings
        threshold = 0.5
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = F.cosine_similarity #/ temperature

        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        mask.fill_diagonal_(0)  # Remove self-similarities

        # Compute numerator (positive pairs) and denominator (all pairs)
        scaled_sim = sim_matrix/temperature
        exp_sim = torch.exp(scaled_sim)
        exp_sim = torch.clamp(exp_sim, min=1e-8)  # Ensure no value is too small

        pos_term = torch.log((mask * exp_sim).sum(1) + 1e-8)  
        neg_term = torch.log(exp_sim.sum(1) - exp_sim.diag() + 1e-8)

        # Contrastive loss
        loss = -(pos_term - neg_term).mean()

        # ===== Threshold-Based Accuracy =====
        predicted_similar = (sim_matrix > threshold).float()

        # True Positives: correctly predicted positive pairs
        true_positives = (predicted_similar * mask).sum()
        false_negatives = ((1 - predicted_similar) * mask).sum()  # Missed positives
        pos_acc = true_positives / (true_positives + false_negatives) * 100.0  # Avoid divide-by-zero

        # True Negatives: correctly predicted negative pairs
        true_negatives = ((1 - predicted_similar) * (1 - mask)).sum()
        false_positives = (predicted_similar * (1 - mask)).sum()  # Wrongly predicted negatives as positives
        neg_acc = true_negatives / (true_negatives + false_positives) * 100.0

        # Total Accuracy
        total_pairs = mask.numel()
        correct_predictions = true_positives + true_negatives
        acc = (correct_predictions / total_pairs) * 100.0

        # Compute mean positive & negative similarity scores
        pos_scores = sim_matrix[mask.bool()].mean()
        neg_scores = sim_matrix[~mask.bool()].mean()

        # Log to wandb
        wandb.log({
            "pos_score": pos_scores.item(),
            "neg_score": neg_scores.item(),
            "accuracy": acc,
            "positive_accuracy": pos_acc.item(),
            "negative_accuracy": neg_acc.item()
        })

        return loss, acc

    def load_model_from_ckpt(self, ckpt_path, log=True):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('model'):
                    base_ckpt[k[len('model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('cls_head_finetune'):
                    del base_ckpt[k]

            keys_to_remove = [
                'embed.first_conv.0.weight',
                'decoder.increase_dim.0.weight',
                'decoder.increase_dim.0.bias'
            ]
            for key in keys_to_remove:
                if key in base_ckpt:
                    del base_ckpt[key]


            incompatible = self.load_state_dict(base_ckpt, strict=False)
            if log:
                if incompatible.missing_keys:
                    print_log('missing_keys', logger='PointTransformer')
                    print_log(
                        get_missing_parameters_message(incompatible.missing_keys),
                        logger='PointTransformer'
                    )
                if incompatible.unexpected_keys:
                    print_log('unexpected_keys', logger='PointTransformer')
                    print_log(
                        get_unexpected_parameters_message(incompatible.unexpected_keys),
                        logger='PointTransformer'
                    )

                print_log(f'[PointTransformer] Successful Loading the ckpt from {ckpt_path}', logger='PointTransformer')
        else:
            print_log('Training from scratch!!!', logger='PointTransformer')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.embed(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()

        query = self.global_query.expand(batch_size, -1, -1)
        
        relative_position = center[:, 1:, :] - center[:, :-1, :]
        relative_norm = torch.norm(relative_position, dim=-1, keepdim=True)
        relative_direction = relative_position / (relative_norm + 1e-5)
        position = torch.cat([center[:, 0, :].unsqueeze(1), relative_direction], dim=1)
        pos_relative = self.pos_embed(position).to(group_input_tokens.dtype)

        pos = self.pos_embed(center).to(group_input_tokens.dtype)

        attn_mask = torch.full(
            (seq_len, seq_len), -float("Inf"), device=group_input_tokens.device, dtype=group_input_tokens.dtype
        ).to(torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        # transformer
        encoded_features, global_tokens = self.encoder(group_input_tokens, pos, attn_mask, query)
        generated_points = self.decoder(encoded_features, pos_relative, attn_mask)

        # neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] + center.unsqueeze(2)
        gt_points = neighborhood.reshape(batch_size * self.num_group, self.group_size, self.input_channel)

        generated_xyz = generated_points[:, :, :3]
        gt_xyz = gt_points[:, :, :3]
        dist1, dist2, idx = self.cd_loss(generated_xyz, gt_xyz)

        cd_l2_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        cd_l1_loss = (torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))) / 2

        img_token = global_tokens[:, :self.img_queries]
        text_token = global_tokens[:, self.img_queries:-1]
   

        point_cloud_embedding = torch.mean(encoded_features, dim=1)
        ret = self.embedding_head(point_cloud_embedding)

        return ret, cd_l1_loss + cd_l2_loss
