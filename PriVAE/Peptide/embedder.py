import torch
from transformers import EsmModel, EsmTokenizer

# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig

class Embedder:
    def __init__(self):
        # model_name = "facebook/esm2_t6_8M_UR50D"
        # # model_name = "facebook/esm2_t33_650M_UR50D"
        # self.tokenizer = EsmTokenizer.from_pretrained(model_name, max_length=30)
        # self.model = EsmModel.from_pretrained(model_name)
        # # print("len    ", len(self.tokenizer))
        # # # print("-----    ", self.tokenizer.image_token_id)
        # # # kkk += 4
        # ####################################
        # # self.client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"
        #############
        pass
    
    def set_max_length(self, mx):
        self.max_length = mx
    
    def set_models(self):
        model_name = "facebook/esm2_t6_8M_UR50D"
        self.tokenizer = EsmTokenizer.from_pretrained(model_name, max_length=self.max_length)
        self.model = EsmModel.from_pretrained(model_name)

    def encode(self, x, max_length=30):
        # x = ''.join(x)
        # print("x     ", x, " -----   ", len(x))
        #####################
        # inputs = self.tokenizer(x, return_tensors="pt")#, padding=True, truncation=True)
        inputs = self.tokenizer(x, return_tensors="pt", max_length=max_length)
        # print("inputs    ", inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print("outputs    ", outputs)
        embeddings = outputs.last_hidden_state
        embeddings = embeddings.squeeze(0)
        pooler = outputs.pooler_output
        # print("emb    ", embeddings.shape)
        # print("pooler     ", pooler.shape)
        #######################
        # protein = ESMProtein(sequence=x)
        # protein_tensor = self.client.encode(protein)
        # logits_output = self.client.logits(
        #     protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        # )
        # print("prot    ", protein_tensor)
        # # print("logits log   ", logits_output.logits)
        # print("logits embed   ", logits_output.embeddings.shape)
        #######################
        # return torch.rand((30, 20))
        return embeddings
        # return pooler

    def decode(self, x):
        pass

    