from app import app

client = app.test_client()
response = client.get('/attendance')
body = response.get_data(as_text=True)
print('status_code=', response.status_code)
print('location=', response.headers.get('Location'))
print('has_video=', 'video' in body.lower())
print('has_title=', 'Face Recognition Attendance' in body)
assert response.status_code == 200, response.status_code
assert response.headers.get('Location') is None, response.headers.get('Location')
assert 'video' in body.lower(), body[:200]
print('VERIFY_OK')
