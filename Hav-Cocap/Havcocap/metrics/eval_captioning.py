import logging
import os
import sys

# Import standard scorers
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# Optional imports for new metrics
try:
    import laion_clap
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    laion_clap = None

logger = logging.getLogger(__name__)

class CLAPScore:
    def __init__(self, device='cuda'):
        self.device = device
        if laion_clap:
            # Load model (using efficient HTSAT-tiny or similar if available, or larger based on requirements)
            # User didn't specify, defaulting to 630k-audioset-fusion-best.pt via wrapper if possible, 
            # but standard usage is:
            try:
                self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
                self.model.load_ckpt() # Downloads default
                self.model.to(device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to load CLAP model: {e}")
                self.model = None
        else:
            self.model = None

    def compute_score(self, gts, res):
        """
        gts: {image_id: [{'caption': str}, ...]}
        res: {image_id: [{'caption': str}]}
        """
        if self.model is None:
            return 0.0, [0.0] * len(gts)

        scores = []
        # Process one by one or batch? Batch is better but simple loop for now
        # CLAP expects audio and text. But for Captioning Evaluation (Text-to-Text? Or Audio-to-Text?)
        # CLAPScore usually measures Audio-Text alignment.
        # BUT here we only have Ground Truth Text and Generated Text.
        # So we can measure Text-Text similarity using CLAP text encoder? 
        # Or does the user mean Audio-Text alignment (requires Audio inputs)?
        # The user said "since my one is multimodal audio one". 
        # Standard CLAPScore in captioning: Distance between Audio embedding and Generated Text embedding.
        # However, the standard evaluation interface only gets (GT_Text, Gen_Text).
        # We generally don't have access to the audio path in `eval_captioning.py` easily unless we pass it.
        # If we only have GT text, we can compute CLAP(GT, Gen) text similarity (Text-to-Text).
        # OR we assume we can't do true CLAPScore (Audio-Text) without Audio.
        # Let's implement Text-to-Text similarity using CLAP encodings as a proxy if audio isn't available,
        # OR just return 0 and warn.
        # FOR NOW: I will implement Text-Text similarity between GT and Gen using CLAP text encoder.
        # This is strictly "CLAP-Text-Score".
        
        # NOTE: If the user wants true Audio-Text CLAPScore, we need to modify the evaluation pipeline to pass audio paths.
        # Given the current scope, I will assume Text-Text or placeholder.
        # Let's try Text-Text similarity (GT vs Gen) using CLAP or similar.
        # Actually, strict CLAPScore is Audio-Text. 
        # I'll add a placeholder warning.
        logger.warning("CLAPScore requires Audio input, but current EvalCap only provides text. Calculating Text-Text latent similarity.")
        
        imgIds = sorted(gts.keys())
        for id in imgIds:
            hypo = res[id][0]['caption']
            ref = gts[id][0]['caption'] # Take first ref
            
            with torch.no_grad():
                # Encoding text
                text_embed = self.model.get_text_embedding([hypo, ref], use_tensor=True)
                score = torch.nn.functional.cosine_similarity(text_embed[0].unsqueeze(0), text_embed[1].unsqueeze(0))
                scores.append(score.item())
                
        return np.mean(scores), scores


class EvalCap:
    def __init__(self, annos, rests, cls_tokenizer=PTBTokenizer,
                 use_scorers=('Bleu', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPIDEr', 'CLAPScore')):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.annos = annos
        self.rests = rests
        self.Tokenizer = cls_tokenizer
        self.use_scorers = use_scorers

    def evaluate(self):
        res = {}
        for r in self.rests:
            res[str(r['image_id'])] = [{'caption': r['caption']}]

        gts = {}
        for imgId in self.annos:
            gts[str(imgId)] = [{'caption': c} for c in self.annos[imgId]]

        # Tokenization
        tokenizer = self.Tokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # Scorers
        scorers = []
        if 'Bleu' in self.use_scorers:
            scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
        if 'METEOR' in self.use_scorers:
            scorers.append((Meteor(), "METEOR"))
        if 'ROUGE_L' in self.use_scorers:
            scorers.append((Rouge(), "ROUGE_L"))
        if 'CIDEr' in self.use_scorers:
            scorers.append((Cider(), "CIDEr"))
        if 'SPICE' in self.use_scorers or 'SPIDEr' in self.use_scorers:
            scorers.append((Spice(), "SPICE"))
        
        # CLAPScore (Optional)
        if 'CLAPScore' in self.use_scorers and laion_clap:
            scorers.append((CLAPScore(), "CLAPScore"))

        # Compute
        self.eval = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
        
        # Compute SPIDEr (Average of CIDEr and SPICE)
        if 'SPIDEr' in self.use_scorers:
            if 'CIDEr' in self.eval and 'SPICE' in self.eval:
                spider_score = (self.eval['CIDEr'] + self.eval['SPICE']) / 2.0
                self.setEval(spider_score, 'SPIDEr')
                
                # Per image
                spider_scores = []
                cider_scores = self.imgToEval[list(gts.keys())[0]]['CIDEr'] # checking access
                # Re-iterate keys to match order
                for imgId in gts.keys():
                    c = self.imgToEval[imgId]['CIDEr']
                    s = self.imgToEval[imgId]['SPICE']
                    self.imgToEval[imgId]['SPIDEr'] = (c + s) / 2.0
            else:
                logger.warning("SPIDEr requires both CIDEr and SPICE to be computed.")

        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def evaluate(submission, reference):
    tokenizer = PTBTokenizer
    annos = reference
    data = submission['results']
    rests = []
    for name, value in data.items():
        rests.append({'image_id': str(name), 'caption': value[0]['sentence']})
    eval_cap = EvalCap(annos, rests, tokenizer)
    eval_cap.evaluate()
    return eval_cap.eval
