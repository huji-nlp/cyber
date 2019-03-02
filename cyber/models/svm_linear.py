from allennlp.data import Vocabulary
from allennlp.models.model import Model
from sklearn.svm import SVC

from cyber.models.svm import Svm


@Model.register("svm_linear")
class SvmLinear(Svm):

    def __init__(self, vocab: Vocabulary) -> None:
        super(Svm, self).__init__(vocab)
        self.svm = SVC(kernel="linear", gamma='scale', cache_size=4000, max_iter=-1)

