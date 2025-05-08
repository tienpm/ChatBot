import fire
import sentencepiece as spm


def test_tokenizer_model(sp_model_name: str = "vi-tokenizer-10k"):
    # Load the trained SentencePiece model
    text = (
        "Ronaldo đảm nhận băng đội trưởng của đội tuyển quốc gia vào tháng 7 năm 2008."
    )
    text += " Năm 2015, Ronaldo được Liên đoàn bóng đá Bồ Đào Nha bầu chọn là cầu thủ Bồ Đào Nha xuất sắc nhất mọi thời đại."

    sp = spm.SentencePieceProcessor()
    model_file = sp_model_name + ".model"
    sp.load(model_file)
    print(sp.encode_as_pieces(text))


if __name__ == "__main__":
    fire.Fire()
