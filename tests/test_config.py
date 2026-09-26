"""配置持久化、隔离、导入校验与凭据来源。全部使用假凭据。"""

import io
import threading
from unittest.mock import Mock

import pytest

import app as app_module
import config
from tasks import TaskManager
from test_app import ORIGIN, establish_session, submit_tts, wait_task


@pytest.fixture
def client(tmp_path):
    manager = TaskManager(tmp_path / "tasks")
    app = app_module.create_app(manager)
    app.testing = True
    try:
        yield app.test_client()
    finally:
        manager.close()


def test_roundtrip_restart_and_permissions(client, tmp_path):
    establish_session(client)
    response = client.get('/config')
    assert response.headers['Cache-Control'] == 'no-store'
    data = response.json
    data['secrets']['google_api_key'] = 'fake-local-key'
    data['tts_languages'].append({'id': 'es', 'label': '西班牙语'})
    data['tts_models']['spanish'] = {'type': 'gtts', 'label': '西班牙语', 'lang': 'es', 'tld': 'com'}
    data['tts_voice_matrix']['gtts']['es'] = {'male': 'spanish', 'female': 'spanish'}
    saved = client.post('/config', json=data, headers=ORIGIN)
    assert saved.status_code == 200
    assert 'es' in saved.json['tts_voice_matrix']['gtts']
    assert client.get('/config').json == data
    assert config.ConfigStore(tmp_path / 'config.json').current.raw == data
    assert (tmp_path / 'config.json').stat().st_mode & 0o777 == 0o600
    assert b'fake-local-key' not in client.get('/').data
    assert b'fake-local-key' not in saved.data


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(speed=20),
    lambda d: d.update(voice_gender='female'),
    lambda d: d['tts_models']['SG-man'].update(speed=-15),
    lambda d: d['tts_models']['SG-man'].update(type=[]),
    lambda d: d['tts_models']['SG-man'].update(service_region=[]),
    lambda d: d['tts_voice_matrix'].pop('azure'),
    lambda d: d['tts_voice_matrix']['azure']['sg-en'].update(male='missing'),
    lambda d: d['secrets'].update(google_cloud_tts_credentials_json='{"type":"service_account"}'),
    lambda d: d['secrets'].update(google_cloud_tts_credentials_json={'type': 'service_account'}),
    lambda d: d['secrets'].update(azure_speech_region='evil.example/path'),
])
def test_invalid_import_preserves_previous(client, tmp_path, mutate):
    establish_session(client)
    original = client.get('/config').json
    assert client.post('/config', json=original, headers=ORIGIN).status_code == 200
    before = (tmp_path / 'config.json').read_bytes()
    mutate(original)
    response = client.post('/config', json=original, headers=ORIGIN)
    assert response.status_code == 400
    assert response.json['error']['code'] == 'invalid_config'
    assert (tmp_path / 'config.json').read_bytes() == before
    assert client.get('/config').json == config.build_default_config()


def test_write_failure_does_not_apply(client, monkeypatch):
    establish_session(client)
    data = client.get('/config').json
    data['ocr_model'] = 'changed'
    monkeypatch.setattr(config.os, 'replace', Mock(side_effect=OSError('disk full')))
    assert client.post('/config', json=data, headers=ORIGIN).status_code == 500
    assert client.get('/config').json['ocr_model'] == config.build_default_config()['ocr_model']


def test_config_same_origin_and_session(client):
    assert client.get('/config').status_code == 403
    establish_session(client)
    assert client.post('/config', json=config.build_default_config()).status_code == 403
    assert client.post('/config', json={}, headers={'Origin': 'https://evil.example'}).status_code == 403
    assert client.post('/config', data='{bad', content_type='application/json', headers=ORIGIN).status_code == 400


def test_missing_credentials_never_use_environment(monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY', 'do-not-use')
    monkeypatch.setenv('GOOGLE_APPLICATION_CREDENTIALS', '/do-not-read.json')
    gemini = Mock()
    cloud = Mock()
    monkeypatch.setattr(app_module.genai, 'Client', gemini)
    monkeypatch.setattr(app_module.texttospeech, 'TextToSpeechClient', cloud)
    runtime = config.RuntimeConfig(config.build_default_config())
    with pytest.raises(ValueError, match='Gemini'):
        app_module.get_gemini_client(runtime)
    with pytest.raises(ValueError, match='Google Cloud'):
        app_module.get_google_cloud_tts_client(runtime.secrets)
    gemini.assert_not_called()
    cloud.assert_not_called()


def test_queued_tasks_keep_submission_configuration(client, monkeypatch):
    establish_session(client)
    old = client.get('/config').json
    old['secrets']['azure_speech_key'] = 'old-fake'
    old['ocr_model'] = 'old-model'
    client.post('/config', json=old, headers=ORIGIN)
    started, release = threading.Event(), threading.Event()
    seen = []

    def synth(text, model, speed, path):
        seen.append(model['_secrets']['azure_speech_key'])
        started.set()
        assert release.wait(5)
        path.write_bytes(b'fake-audio')

    def extract(path, mime, runtime):
        seen.append(runtime.ocr_model)
        return app_module.parse_ocr_response('Title\nHello.')

    monkeypatch.setattr(app_module, 'synthesize_item', synth)
    monkeypatch.setattr(app_module, 'extract_text_cloud', extract)
    first = submit_tts(client, [{'text': 'Hello.', 'bold_words': [{'word': 'Hello'}]}]).json
    assert started.wait(2)
    queued = client.post('/upload', data={'file': (io.BytesIO(b'raw'), 'image.jpg')}, headers=ORIGIN).json
    try:
        old['secrets']['azure_speech_key'] = 'new-fake'
        old['ocr_model'] = 'new-model'
        assert client.post('/config', json=old, headers=ORIGIN).status_code == 200
    finally:
        release.set()
    assert wait_task(client, first['task_id'])['status'] == 'succeeded'
    assert wait_task(client, queued['task_id'])['status'] == 'succeeded'
    assert seen == ['old-fake', 'old-fake', 'old-model']
    next_task = submit_tts(client, [{'text': 'Hi.'}]).json
    wait_task(client, next_task['task_id'])
    assert seen[-1] == 'new-fake'


def test_bad_file_fails_startup(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{broken')
    with pytest.raises(ValueError):
        config.ConfigStore(path)


def test_provider_calls_use_configured_credentials(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    data = config.build_default_config()
    data['secrets']['google_api_key'] = 'fake-from-file'
    runtime = config.RuntimeConfig(data)
    gemini = Mock()
    monkeypatch.setenv('GOOGLE_API_KEY', 'ignored-env')
    monkeypatch.setattr(app_module.genai, 'Client', gemini)
    app_module.get_gemini_client(runtime)
    assert gemini.call_args.kwargs['api_key'] == 'fake-from-file'
    assert gemini.call_args.kwargs['vertexai'] is False

    azure = Mock(return_value=Mock(content=b'azure-audio'))
    monkeypatch.setattr(app_module.requests, 'post', azure)
    model = runtime.tts_models['SG-woman']
    model['_secrets'] = {'azure_speech_key': 'fake-azure', 'azure_speech_region': 'eastus'}
    monkeypatch.setenv('AZURE_API_KEY', 'ignored-env')
    app_module.synthesize_azure_rest('Hello', model, -15, tmp_path / 'azure.mp3')
    assert azure.call_args.args[0] == 'https://eastus.tts.speech.microsoft.com/cognitiveservices/v1'
    assert azure.call_args.kwargs['headers']['Ocp-Apim-Subscription-Key'] == 'fake-azure'

    client = MagicMock()
    client.__enter__.return_value = client
    client.synthesize_speech.return_value.audio_content = b'cloud-audio'
    credentials = Mock(return_value='explicit-credentials')
    constructor = Mock(return_value=client)
    monkeypatch.setattr(app_module.service_account.Credentials, 'from_service_account_info', credentials)
    monkeypatch.setattr(app_module.texttospeech, 'TextToSpeechClient', constructor)
    model = runtime.tts_models['UK-Chirp-man']
    model['_secrets'] = {'google_cloud_tts_credentials_json': {'fake': 'service-account'}}
    app_module.synthesize_google_cloud('Hello', model, 15, tmp_path / 'cloud.mp3')
    credentials.assert_called_once_with({'fake': 'service-account'})
    constructor.assert_called_once_with(credentials='explicit-credentials')
    assert client.synthesize_speech.call_args.kwargs['retry'] is None
    assert client.synthesize_speech.call_args.kwargs['timeout'] == 30
    client.__exit__.assert_called_once()


@pytest.mark.parametrize('peer,allowed', [
    ('127.0.0.1', True), ('::1', True), ('192.168.0.42', True),
    ('100.64.0.1', True), ('100.127.255.254', True),
    ('::ffff:192.168.0.42', True),
    ('8.8.8.8', False), ('192.168.1.42', False), ('10.0.0.1', False),
    ('172.17.0.1', False), ('100.128.0.1', False), ('2001:db8::1', False),
    ('::ffff:8.8.8.8', False), ('', False), ('invalid', False),
])
def test_config_network_boundary(client, peer, allowed):
    env = {'REMOTE_ADDR': peer}
    page = client.get('/', environ_overrides=env)
    assert page.status_code == 200
    assert (b'id="config-btn"' in page.data) is allowed
    assert (b'id="config-dialog"' in page.data) is allowed
    assert page.headers['Cache-Control'] == 'no-store'
    read = client.get('/config', environ_overrides=env)
    write = client.post('/config', json=config.build_default_config(), headers=ORIGIN, environ_overrides=env)
    assert read.status_code == write.status_code == (200 if allowed else 403)
    if not allowed:
        assert read.json['error']['code'] == 'config_forbidden'
        assert 'secrets' not in read.json
        assert client.get('/health', environ_overrides=env).status_code == 200


@pytest.mark.parametrize('header', app_module.CONFIG_PROXY_HEADERS)
@pytest.mark.parametrize('peer', ['127.0.0.1', '192.168.0.42', '8.8.8.8'])
def test_proxy_headers_cannot_grant_config_access(client, header, peer):
    env = {'REMOTE_ADDR': peer}
    headers = {**ORIGIN, header: '192.168.0.42'}
    page = client.get('/', headers=headers, environ_overrides=env)
    assert b'id="config-btn"' not in page.data
    assert client.get('/config', headers=headers, environ_overrides=env).status_code == 403
    assert client.post('/config', json=config.build_default_config(), headers=headers, environ_overrides=env).status_code == 403
    # Even empty Cloudflare/proxy markers must fail closed.
    assert client.get('/config', headers={header: ''}, environ_overrides=env).status_code == 403


def test_default_file_has_no_credentials_and_runtime_takes_precedence(tmp_path):
    default = config.build_default_config()
    assert all(default['secrets'][key] is None for key in config.SECRET_KEYS if key != 'azure_speech_region')
    assert config.validate_config(default) == default
    store = config.ConfigStore(tmp_path / 'config.json')
    assert store.current.raw == default
    original = config.DEFAULT_CONFIG_PATH.read_bytes()
    data = store.current.raw
    data['secrets']['google_api_key'] = 'fake-runtime-only'
    store.save(data)
    assert config.ConfigStore(store.path).current.secrets['google_api_key'] == 'fake-runtime-only'
    assert config.DEFAULT_CONFIG_PATH.read_bytes() == original


def test_default_file_rejects_credentials(tmp_path, monkeypatch):
    import json
    data = config.build_default_config()
    data['secrets']['google_api_key'] = 'do-not-commit'
    path = tmp_path / 'config.default.json'
    path.write_text(json.dumps(data))
    monkeypatch.setattr(config, 'DEFAULT_CONFIG_PATH', path)
    with pytest.raises(config.ConfigError, match='不允许包含密钥'):
        config.build_default_config()


def test_upgrade_defaults_does_not_change_saved_config(tmp_path, monkeypatch):
    import json
    saved = config.build_default_config()
    saved['secrets']['google_api_key'] = 'fake-saved-key'
    saved['ocr_model'] = 'user-selected-model'
    saved['tts_languages'][0]['label'] = '用户自定义语言名称'
    path = tmp_path / 'data' / 'config.json'
    config.ConfigStore(path).save(saved)
    before = path.read_bytes()
    key = config.session_key(path.parent)

    updated = config.build_default_config()
    updated['ocr_model'] = 'new-release-default-model'
    updated['tts_languages'][0]['label'] = '新版默认名称'
    defaults = tmp_path / 'new-release' / 'config.default.json'
    defaults.parent.mkdir()
    defaults.write_text(json.dumps(updated))
    monkeypatch.setattr(config, 'DEFAULT_CONFIG_PATH', defaults)

    # 模拟更新后启动：仍挂载原数据目录，启动清理不能触及配置和签名密钥。
    manager = app_module.bootstrap_runtime(path.parent)
    try:
        app = app_module.create_app(manager)
        assert app.extensions['config_store'].current.raw == saved
        assert app.secret_key == key
        assert path.read_bytes() == before
    finally:
        manager.close()
    assert config.ConfigStore(tmp_path / 'fresh-install' / 'config.json').current.raw == updated


def test_saved_config_loading_does_not_read_defaults(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    saved = config.build_default_config()
    config.ConfigStore(path).save(saved)
    monkeypatch.setattr(config, 'build_default_config', Mock(side_effect=AssertionError('default read')))
    assert config.ConfigStore(path).current.raw == saved


def test_incompatible_saved_config_is_not_replaced_by_defaults(tmp_path):
    import json
    saved = config.build_default_config()
    saved['version'] = 999
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    with pytest.raises(config.ConfigError, match='不支持的配置版本'):
        config.ConfigStore(path)
    assert path.read_bytes() == before


def test_friendly_config_form_is_separate_from_upload(client):
    page = client.get('/').data
    assert b'/static/config-editor.js' in page
    assert b'id="config-engine"' in page
    assert b'id="config-add-language"' in page
    assert b'id="config-region"' not in page
    assert b'id="config-catalog"' not in page
    assert client.get('/static/config-editor.js').status_code == 200
    public = client.get('/', headers={'CF-Connecting-IP': '8.8.8.8'}).data
    assert b'/static/config-editor.js' not in public
    assert b'id="select-file-btn"' in public


def test_preferred_selection_persists_and_initializes_home(client, tmp_path):
    import json
    import re
    establish_session(client)
    data = client.get('/config').json
    data['preferred_tts'] = {'engine': 'gtts', 'language': 'fr-fr'}
    response = client.post('/config', json=data, headers=ORIGIN)
    assert response.status_code == 200
    assert response.json['preferred_tts'] == data['preferred_tts']
    assert client.get('/config').json['preferred_tts'] == data['preferred_tts']
    assert config.ConfigStore(tmp_path / 'config.json').current.preferred_tts == data['preferred_tts']
    home = client.get('/').get_data(as_text=True)
    selected = json.loads(re.search(r'const defaultTtsSelection = (.*);', home).group(1))
    assert selected == data['preferred_tts']


@pytest.mark.parametrize('preferred', [
    None, {}, {'engine': [], 'language': 'sg-en'},
    {'engine': 'missing', 'language': 'sg-en'},
    {'engine': 'google', 'language': 'sg-en'},
    {'engine': 'azure', 'language': 'sg-en', 'gender': 'female'},
    {'engine': 'azure', 'language': 'sg-en', 'speed': 10},
])
def test_invalid_preference_does_not_replace_saved_config(client, preferred):
    establish_session(client)
    original = client.get('/config').json
    data = {**original, 'preferred_tts': preferred}
    assert client.post('/config', json=data, headers=ORIGIN).status_code == 400
    assert client.get('/config').json == original


def test_old_configuration_has_default_without_rewrite(tmp_path):
    import json
    data = config.build_default_config()
    del data['preferred_tts']
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    loaded = config.ConfigStore(path).current
    assert loaded.preferred_tts == {'engine': 'azure', 'language': 'sg-en'}
    assert 'preferred_tts' not in loaded.raw
    assert path.read_bytes() == before
    del data['tts_voice_matrix']['azure']
    data['tts_engines'] = [e for e in data['tts_engines'] if e['id'] != 'azure']
    fallback = config.RuntimeConfig(data).preferred_tts
    assert fallback['engine'] == data['tts_engines'][0]['id']
    assert fallback['language'] in data['tts_voice_matrix'][fallback['engine']]
