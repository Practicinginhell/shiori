// Shapes of the JSON the Python server sends (tanklight/simulate.py: Community.snapshot).

export type StateName = "protected" | "check" | "boil" | "service";
export type Lang = "en" | "fr" | "iu";

export interface Status {
  state: StateName;
  raw_state: StateName;
  reasons: string[];
  faults: string[];
  fc_mg_l: number | null;
  hours_protected: number | null;
  level_pct: number | null;
  days_left: number | null;
  essential_days: number | null;
  runout_risk: number | null;
}

export interface SewageStatus {
  state: "ok" | "soon" | "urgent" | "service";
  reasons: string[];
  level_pct: number | null;
  days_to_full: number | null;
  temp_c: number | null;
}

export interface PumpOutRow {
  home_id: string;
  tier: number;
  why: string;
  can_skip: boolean;
  level_pct: number | null;
  days_to_full: number | null;
  people: number | null;
}

export interface TimelinePoint {
  h: number;
  fc_true: number;
  fc_est: number | null;
  level_pct: number;
  state: StateName;
}

export interface HomeRow {
  home_id: string;
  people: number;
  capacity_l: number;
  state: StateName | null;
  reasons: string[] | null;
  level_pct: number | null;
  days_left: number | null;
  runout_risk: number | null;
}

export interface FleetRow {
  home_id: string;
  tier: number;
  why: string;
  can_skip: boolean;
  level_pct: number | null;
  days_left: number | null;
  people: number | null;
}

export interface Stats {
  runouts: number;
  dry_home_hours: number;
  pump_outs: number;
  sewage_backup_hours: number;
  visits: number;
  litres_delivered: number;
}

export interface Snapshot {
  simulated: boolean;
  hour: number;
  day: number;
  clock: string;
  season: "january" | "july";
  season_label: string;
  storm_today: boolean;
  policy: "fixed" | "priority";
  visits_per_day: number;
  featured: {
    home_id: string;
    people: number;
    capacity_l: number;
    probe_attached: boolean;
    engine: "graphflow" | "direct";
    status: Status;
    timeline: TimelinePoint[];
    sewage: SewageStatus;
    heater_on: boolean;
  };
  homes: HomeRow[];
  fleet: FleetRow[];
  pump_outs: PumpOutRow[];
  inbox: InboxDecision[];
  pump_outs_per_day: number;
  stats: Stats;
}

export interface InboxDecision {
  text: string;
  home_id: string;
  needs: string[];
  route: "water_quality" | "pump_out" | "water_delivery" | "device" | "person";
  urgent: boolean;
  action: string;
  probabilities: Record<string, number>;
  model: string | null;
  error: string | null;
}

export interface OfficePlan {
  scores: Record<string, number>;
  model: string;
  predicted: number[];
  timesfm_installed: boolean;
  model_error: string | null;
  note: string;
  source: "llm" | "template";
  draft: string | null;
  writer_error: string | null;
  llm: string | null;
}
