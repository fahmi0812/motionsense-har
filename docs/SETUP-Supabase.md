# SETUP Supabase (Gratis)
## 1. Buat Project
1. https://supabase.com → Sign in (GitHub) → **New project**.
2. Nama `gerak-kenal`, region **Southeast Asia (Singapore)**, simpan password DB.
3. Tunggu ±2 menit.

## 2. Buat Tabel (SQL Editor → New query → Run)
```sql
create table if not exists public.sensor_readings (
  id            bigserial primary key,
  user_id       text        not null,
  session_id    uuid,
  label         text        not null check (label in
                 ('WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING')),
  ts            timestamptz not null,
  ax double precision, ay double precision, az double precision,
  gx double precision, gy double precision, gz double precision,
  sampling_rate int         not null default 50,
  sumber        text        not null default 'privat',
  created_at    timestamptz not null default now(),
  unique (session_id, ts)            -- cegah duplikat saat re-sync
);
create index if not exists idx_sr_user_label on public.sensor_readings (user_id, label);
create index if not exists idx_sr_session_ts on public.sensor_readings (session_id, ts);

-- Ringkasan cepat untuk cek progres 25%
create or replace view public.v_progress as
select label, count(*) as sampel, round(count(*)/50.0/60,1) as menit,
       floor(count(*)/64.0) as perkiraan_window
from public.sensor_readings group by label order by label;
```

## 3. Row Level Security (dasar)
```sql
alter table public.sensor_readings enable row level security;
-- APK (anon) hanya boleh INSERT, tidak bisa baca/hapus
create policy "anon insert" on public.sensor_readings
  for insert to anon with check (sumber = 'privat');
```
Baca/export dilakukan dari dashboard atau pakai service key (server saja).

## 4. Ambil Kredensial
Project Settings → **API**:
- `Project URL` → `SUPABASE_URL`
- `anon public` → `SUPABASE_ANON_KEY` (boleh di APK, dilindungi RLS)
- `service_role` → **RAHASIA**, hanya untuk server/Vercel/laptop. Jangan pernah di APK.

> ⚠️ **API key JANGAN di-hardcode atau di-commit ke repo publik.** Simpan di env/secret:
```bash
flutter run --dart-define=SUPABASE_URL=https://xxxx.supabase.co --dart-define=SUPABASE_ANON_KEY=...
```
Tambahkan `.env`, `*.keystore`, `env.json` ke `.gitignore`.

## 5. Tes Insert (curl)
```bash
curl -X POST "$SUPABASE_URL/rest/v1/sensor_readings" \
 -H "apikey: $SUPABASE_ANON_KEY" -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
 -H "Content-Type: application/json" -H "Prefer: return=minimal" \
 -d '[{"user_id":"tes","label":"SITTING","ts":"2026-10-28T00:00:00Z","ax":0,"ay":9.8,"az":0,"gx":0,"gy":0,"gz":0}]'
```
Harus `201`. Hapus data tes: `delete from sensor_readings where user_id='tes';`

## 6. Export ke CSV
- **Dashboard:** Table Editor → `sensor_readings` → Export → CSV (cocok < 100rb baris).
- **Data besar (disarankan):** SQL Editor tidak cocok; pakai psql:
```bash
psql "$SUPABASE_DB_URL" -c "\copy (select user_id,session_id,label,ts,ax,ay,az,gx,gy,gz,sampling_rate,sumber from sensor_readings order by session_id,ts) to 'sensor_readings.csv' csv header"
```
Simpan ke `zip2/data/privat/sensor_readings.csv`.

## 7. Batas Free Tier
DB 500 MB ≈ ±3–4 juta baris — cukup (target ±650rb baris). Project pause setelah 7 hari tanpa aktivitas → buka dashboard untuk restore.

## 6. Tambahan untuk Web MotionSense (izin BACA)
Web perlu membaca data. Pilih SATU:
- **A (disarankan):** isi `SUPABASE_SERVICE_KEY` di `.env` server (Render/HF/laptop). Jangan pernah taruh di frontend/APK.
- **B (demo kelas, data tidak sensitif):** izinkan anon membaca:
```sql
create policy "anon read" on public.sensor_readings for select to anon using (true);
```
Catatan: di tabel ini nama kolom label adalah `label`, ada juga `user_id` dan `sumber`.
