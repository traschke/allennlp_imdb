from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('imdb')
class ImdbPredictor(Predictor):
    """Predictor wrapper for the AcademicPaperClassifier"""
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        review = json_dict['review']
        instance = self._dataset_reader.text_to_instance(string=review, label=0)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        # label_dict = self._model.vocab.get_index_to_token_vocabulary('label')
        # Convert it to list ["ACL", "AI", ...]
        # all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {"instance": self.predict_instance(instance)}