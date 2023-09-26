import zipfile
import io


def zip_file(data: list[tuple[str, str]]):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, j in data:
            zip_file.writestr(i, j)
    return zip_buffer.getvalue()
