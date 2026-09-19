import type {
  FileMapping,
  ImportApplyResult,
  ImportCandidate,
  ImportPreview,
} from '#/manualImport';

export const apiManualImport = {
  async getCandidates() {
    const { data } = await axios.get<ImportCandidate[]>(
      'api/v1/manual-import/candidates',
      { silent: true }
    );
    return data!;
  },

  async preview(torrent_hash: string, official_title: string, season: number) {
    const { data } = await axios.post<ImportPreview>(
      'api/v1/manual-import/preview',
      { torrent_hash, official_title, season }
    );
    return data!;
  },

  async apply(
    torrent_hash: string,
    target_folder: string,
    mappings: FileMapping[]
  ) {
    const { data } = await axios.post<ImportApplyResult[]>(
      'api/v1/manual-import/apply',
      { torrent_hash, target_folder, mappings }
    );
    return data!;
  },
};
