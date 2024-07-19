import csv
from lxml import etree

# XML 파일 경로
xml_file_path = 'apart_code.xml'

try:
    # XML 파일을 파싱
    tree = etree.parse(xml_file_path)
    root = tree.getroot()
    print("XML 파일이 성공적으로 파싱되었습니다.")
except etree.XMLSyntaxError as e:
    print(f"XML 파싱 오류: {e}")
    exit(1)
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {xml_file_path}")
    exit(1)
except Exception as e:
    print(f"기타 오류: {e}")
    exit(1)

# CSV 파일을 작성하기 위한 준비
csv_file_path = 'output.csv'

try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # 헤더 작성
        header = ['as1', 'as2', 'as3', 'bjdCode', 'kaptCode', 'kaptName']
        csv_writer.writerow(header)

        # XML에서 데이터 추출
        items = root.find('.//items')
        if items is not None:
            for item in items.findall('item'):
                row = [
                    item.find('as1').text if item.find('as1') is not None else '',
                    item.find('as2').text if item.find('as2') is not None else '',
                    item.find('as3').text if item.find('as3') is not None else '',
                    item.find('bjdCode').text if item.find('bjdCode') is not None else '',
                    item.find('kaptCode').text if item.find('kaptCode') is not None else '',
                    item.find('kaptName').text if item.find('kaptName') is not None else ''
                ]
                csv_writer.writerow(row)
        else:
            print("items 요소를 찾을 수 없습니다.")
except IOError as e:
    print(f"CSV 파일 작성 오류: {e}")
