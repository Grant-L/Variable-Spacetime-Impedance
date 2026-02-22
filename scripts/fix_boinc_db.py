import xml.etree.ElementTree as ET
import os

print("==> Provisioning MySQL Sub-User Profile...")
os.system("mysql -e \"CREATE USER IF NOT EXISTS 'ave_alpha'@'localhost' IDENTIFIED BY '<YOUR_DB_PASSWORD>'; GRANT ALL PRIVILEGES ON ave_alpha_search.* TO 'ave_alpha'@'localhost'; FLUSH PRIVILEGES;\"")

print("==> Patching BOINC config.xml with new credentials...")
tree = ET.parse('/root/projects/ave_alpha_search/config.xml')
root = tree.getroot()
config = root.find('config')
config.find('db_user').text = 'ave_alpha'
config.find('db_passwd').text = '<YOUR_DB_PASSWORD>'
config.find('db_host').text = 'localhost'
tree.write('/root/projects/ave_alpha_search/config.xml')
print("==> Patch Complete.")
