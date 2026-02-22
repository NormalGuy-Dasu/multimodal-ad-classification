# File: main_app/model_class.py
import torch
import torch.nn as nn
from transformers import CLIPModel, ViltForImagesAndTextClassification
from peft import LoraConfig, get_peft_model

class MultimodalCLIPClassifier(nn.Module):
    def __init__(self, num_classes, base_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(base_model_name, use_safetensors=True)
        
        # Cấu hình LoRA y hệt lúc train
        config = LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.1,
            bias="none",
        )
        self.clip = get_peft_model(self.clip, config)
        
        # Classifier Head
        self.classifier = nn.Linear(self.clip.config.projection_dim * 2, num_classes)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Lưu ý: Model lúc train có chunking (batch, num_chunks, seq_len)
        # Nên đầu vào phải đảm bảo đúng chiều
        batch_size, num_chunks, seq_len = input_ids.shape
        
        # 1. Vision Encoder
        vision_outputs = self.clip.base_model.model.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.base_model.model.visual_projection(vision_outputs[1])

        # 2. Text Encoder
        flat_input_ids = input_ids.view(-1, seq_len) 
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        text_outputs = self.clip.base_model.model.text_model(
            input_ids=flat_input_ids, 
            attention_mask=flat_attention_mask
        )
        text_embeds_flat = self.clip.base_model.model.text_projection(text_outputs[1])
        
        # Reshape lại về (Batch, Chunks, Dim) và lấy trung bình các chunks
        text_embeds = text_embeds_flat.view(batch_size, num_chunks, -1)
        text_embeds = torch.mean(text_embeds, dim=1) 

        # 3. Combine
        combined_features = torch.cat((image_embeds, text_embeds), dim=1)
        logits = self.classifier(combined_features)
        
        return logits
        
# --- 2. MODEL ViLT (MỚI) ---
def get_vilt_model(num_classes, device):
    print("Initialize ViLT Architecture...")
    # Load base model (phải khớp với notebook training)
    model = ViltForImagesAndTextClassification.from_pretrained(
        "dandelin/vilt-b32-mlm",
        num_labels=num_classes,
        id2label={i: str(i) for i in range(num_classes)},
        label2id={str(i): i for i in range(num_classes)},
        num_images=1,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    
    # Cấu hình LoRA (phải khớp với notebook training của bạn)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"] # Quan trọng: ViLT thường train lại classifier head
    )
    
    model = get_peft_model(model, peft_config)
    return model.to(device)