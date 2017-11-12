from Bayes.document import Document

import os

INPUTS_REL = os.path.join("inputs", "pu1")
SPAM_TAG = "spmsg"

root = os.path.abspath(os.path.dirname(__file__))
inputs = os.path.join(root, INPUTS_REL)


def read_buckets():
    buckets = []

    # full path sorted from 0 to 9
    parts = list(map(lambda part: os.path.join(inputs, part), os.listdir(inputs)))
    parts.sort()

    for dir in parts:
        dir_bucket = []

        for doc in os.listdir(dir):
            # full path
            doc_full_path = os.path.join(dir, doc)

            with (open(doc_full_path, 'r')) as file:
                is_spam = (SPAM_TAG in doc_full_path)

                # -1 to get rid of trailing \n
                subject = file.readline()[:-1]
                file.readline()
                content = file.readline()[:-1]

                # get rid of "Subject: " prefix and split
                subject = subject.split(' ', maxsplit=2)[-1]
                subject = [x for x in subject.split(' ') if x]

                content = [x for x in content.split(' ') if x]

                dir_bucket.append(Document(doc, subject, content, is_spam))

        buckets.append(dir_bucket)

    return buckets