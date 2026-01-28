# Log Retention Policy

## Summary
- Production application logs are retained for 30 days.
- Security audit logs are retained for 365 days.
- Debug logs are retained for 7 days.

## Storage Locations
- Application logs: s3://logs/app/
- Security logs: s3://logs/security/

## Access
- Security team has read access to security logs.
- Engineering managers can request temporary access to application logs.

## Exceptions
- Legal holds override the default retention period.
