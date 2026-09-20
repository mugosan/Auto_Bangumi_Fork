export interface ImportCandidate {
  hash: string;
  name: string;
  save_path: string;
  category: string;
  size: number;
  progress: number;
  state: string;
}

export interface FileMapping {
  source_path: string;
  target_path: string;
  episode: number | null;
  parsed: boolean;
  kind: 'media' | 'subtitle';
}

export interface ImportPreview {
  torrent_hash: string;
  official_title: string;
  year: string | null;
  tvdb_id: number | null;
  id_source: string | null;
  target_folder: string;
  mappings: FileMapping[];
  unparsed: string[];
}

export interface ImportApplyResult {
  source_path: string;
  target_path: string;
  succeeded: boolean;
  detail: string | null;
}

export interface FolderFileMapping {
  source_path: string;
  target_path: string;
  episode: number | null;
  parsed: boolean;
  kind: 'media' | 'subtitle';
}

export interface FolderImportPreview {
  source_root: string;
  official_title: string;
  year: string | null;
  tvdb_id: number | null;
  id_source: string | null;
  target_folder: string;
  mappings: FolderFileMapping[];
  unparsed: string[];
}
