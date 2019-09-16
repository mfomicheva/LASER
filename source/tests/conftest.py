import os
import pytest

from source.embed import SentenceEncoder

LASER = os.environ['LASER']


@pytest.fixture
def sentence_encoder():
    enc = SentenceEncoder(
        os.path.join(LASER, 'models/bilstm.93langs.2018-12-26.pt'),
        max_tokens=1200,
        sort_kind='quicksort',
        cpu=True
    )
    return enc
