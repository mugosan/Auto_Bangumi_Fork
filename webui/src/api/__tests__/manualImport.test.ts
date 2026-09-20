/**
 * Contract tests for apiManualImport: call the real functions against a
 * mocked axios instance so a drift between the wrapper and the FastAPI
 * routes in backend/src/module/api/manual_import.py fails a test instead of
 * going unnoticed.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createAxiosMock } from '@/test/mocks/axios';

import { apiManualImport } from '@/api/manualImport';
import { axios } from '@/utils/axios';
import type { ImportCandidate } from '#/manualImport';

vi.mock('@/utils/axios', () => ({ axios: createAxiosMock() }));

const mockCandidate: ImportCandidate = {
  hash: 'abc123',
  name: 'My Show Batch',
  save_path: '/downloads',
  category: '',
  size: 100,
  progress: 0,
  state: 'metaDL',
};

describe('Manual Import API contract (path + HTTP method)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should GET api/v1/manual-import/candidates when listing candidates', async () => {
    (axios.get as any).mockResolvedValue({ data: [mockCandidate] });
    const result = await apiManualImport.getCandidates();
    expect(axios.get).toHaveBeenCalledWith(
      'api/v1/manual-import/candidates',
      { silent: true }
    );
    expect(result).toEqual([mockCandidate]);
  });

  it('should POST api/v1/manual-import/upload with the file as form data', async () => {
    (axios.post as any).mockResolvedValue({ data: mockCandidate });
    const file = new File(['torrent-bytes'], 'show.torrent');
    const result = await apiManualImport.uploadTorrent(file);

    expect(axios.post).toHaveBeenCalledTimes(1);
    const [url, body] = (axios.post as any).mock.calls[0];
    expect(url).toBe('api/v1/manual-import/upload');
    expect(body).toBeInstanceOf(FormData);
    expect((body as FormData).get('file')).toBe(file);
    expect(result).toEqual(mockCandidate);
  });

  it('should POST api/v1/manual-import/preview with torrent_hash/official_title/season', async () => {
    (axios.post as any).mockResolvedValue({
      data: {
        torrent_hash: 'abc123',
        official_title: 'My Show',
        year: '2024',
        tvdb_id: null,
        id_source: null,
        target_folder: '/downloads/My Show',
        mappings: [],
        unparsed: [],
      },
    });
    await apiManualImport.preview('abc123', 'My Show', 1);
    expect(axios.post).toHaveBeenCalledWith('api/v1/manual-import/preview', {
      torrent_hash: 'abc123',
      official_title: 'My Show',
      season: 1,
    });
  });

  it('should POST api/v1/manual-import/folder/preview with path/official_title/season', async () => {
    (axios.post as any).mockResolvedValue({
      data: {
        source_root: '/downloads/incoming',
        official_title: 'My Show',
        year: '2024',
        tvdb_id: null,
        id_source: null,
        target_folder: '/downloads/My Show',
        mappings: [],
        unparsed: [],
      },
    });
    await apiManualImport.previewFolder('/downloads/incoming', 'My Show', 1);
    expect(axios.post).toHaveBeenCalledWith(
      'api/v1/manual-import/folder/preview',
      { path: '/downloads/incoming', official_title: 'My Show', season: 1 }
    );
  });
});
